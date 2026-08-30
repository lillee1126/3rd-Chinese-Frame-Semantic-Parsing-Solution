#!/usr/bin/env python3
# Task1：框架分类训练入口，按开发集准确率选取最佳权重。
# 主方案由运行脚本传入 semantic_weight=1、distance_power=2；下方参数默认值不一定等于运行脚本配置。
# 训练需要外部数据、预训练配置/词表/权重；本代码包不包含这些文件。
import argparse
import json
import math
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertConfig, BertTokenizer, get_linear_schedule_with_warmup

from build_task1_frame_semantics import load_bert_checkpoint
from model_task1_semantic import SemanticAwareFrameLoss, TargetAwareFrameClassifier
from task1_semantic_data import (
    Task1SemanticCollator,
    Task1SemanticDataset,
    collect_seen_frame_ids,
)


# 解析数据路径、模型配置及运行参数；命令行传入值会覆盖这里的默认值。
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", default="./dataset/cfn-train.json")
    parser.add_argument("--dev-file", default="./dataset/cfn-dev.json")
    parser.add_argument("--frame-info", default="./dataset/frame_info.json")
    parser.add_argument("--config-file", default="/root/autodl-tmp/chinese_macbert_large/config.json")
    parser.add_argument("--vocab-file", default="/root/autodl-tmp/chinese_macbert_large/vocab.txt")
    parser.add_argument(
        "--init-checkpoint",
        default="/root/autodl-tmp/chinese_macbert_large/pytorch_model.bin",
    )
    parser.add_argument(
        "--semantic-cache",
        default="./knowledge/task1_frame_semantics_macbert_large.pt",
    )
    parser.add_argument(
        "--output-dir",
        default="./saves/task1_semantic_p2_w1_macbert_large",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--semantic-cache-batch-size", type=int, default=32)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=0,
        help="Stop after this many consecutive epochs without dev accuracy improvement; 0 disables it",
    )
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--head-learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=320)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--semantic-weight", type=float, default=0.2)
    parser.add_argument("--semantic-warmup-epochs", type=int, default=1)
    parser.add_argument("--distance-power", type=float, default=1.0)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument(
        "--class-weighting",
        choices=["none", "sqrt_inv"],
        default="none",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-steps", type=int, default=0)
    parser.add_argument("--max-eval-steps", type=int, default=0)
    parser.add_argument("--rebuild-semantic-cache", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


# 统一设置 Python、NumPy 和 PyTorch 随机种子，减少实验随机差异。
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# 仅将张量迁移到 CPU/GPU，保留句子ID、坐标列表等非张量对象。
def move_batch(batch, device):
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


# 以类别频次的平方根倒数生成权重，提高低频框架在损失中的相对权重。
def build_class_weights(dataset):
    counts = torch.zeros(len(dataset.frame_names), dtype=torch.float)
    for sample in dataset.samples:
        counts[dataset.frame2id[sample["frame"]]] += 1
    weights = 1.0 / torch.sqrt(counts + 1.0)
    return weights / weights.mean()


# 缓存不存在或要求重建时，调用配套语义构建脚本生成缓存。
def ensure_semantic_cache(args):
    cache_path = Path(args.semantic_cache)
    if cache_path.exists() and not args.rebuild_semantic_cache:
        return
    command = [
        sys.executable,
        str(Path(__file__).with_name("build_task1_frame_semantics.py")),
        "--frame-info", args.frame_info,
        "--config-file", args.config_file,
        "--vocab-file", args.vocab_file,
        "--init-checkpoint", args.init_checkpoint,
        "--output", args.semantic_cache,
        "--batch-size", str(args.semantic_cache_batch_size),
    ]
    print("Semantic cache is missing; building it now.")
    subprocess.run(command, check=True)


# 读取语义距离矩阵，核对框架顺序及矩阵形状，避免标签错位。
def load_semantic_distance(path, frame_names):
    cache = torch.load(path, map_location="cpu")
    if cache.get("frame_names") != frame_names:
        raise ValueError(
            "The frame order in the semantic cache differs from frame_info.json. "
            "Run again with --rebuild-semantic-cache."
        )
    distance = cache["distance"].float()
    expected = (len(frame_names), len(frame_names))
    if tuple(distance.shape) != expected:
        raise ValueError(f"Expected semantic distance {expected}, got {distance.shape}")
    return distance


# 按预热轮数逐步增加语义损失权重；预热结束后保持目标权重。
def semantic_weight_for_epoch(epoch_index, target_weight, warmup_epochs):
    if warmup_epochs <= 0:
        return target_weight
    scale = min(1.0, max(0.0, epoch_index / warmup_epochs))
    return target_weight * scale


# 关闭梯度计算，统计总体及已见/未见框架准确率，并记录两类损失。
@torch.no_grad()
def evaluate(model, criterion, loader, device, seen_frame_ids, max_steps=0):
    model.eval()
    total = 0
    correct = 0
    seen_total = 0
    seen_correct = 0
    unseen_total = 0
    unseen_correct = 0
    ce_sum = 0.0
    semantic_sum = 0.0
    seen_tensor = torch.zeros(model.num_labels, dtype=torch.bool, device=device)
    seen_tensor[list(seen_frame_ids)] = True

    for step, raw_batch in enumerate(tqdm(loader, desc="eval")):
        batch = move_batch(raw_batch, device)
        output = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            target_mask=batch["target_mask"],
            content_mask=batch["content_mask"],
        )
        loss_output = criterion(output["logits"], batch["labels"])
        predictions = output["logits"].argmax(dim=-1)
        matches = predictions.eq(batch["labels"])
        batch_size = batch["labels"].shape[0]

        label_seen = seen_tensor[batch["labels"]]
        total += batch_size
        correct += matches.sum().item()
        seen_total += label_seen.sum().item()
        seen_correct += (matches & label_seen).sum().item()
        unseen_total += (~label_seen).sum().item()
        unseen_correct += (matches & ~label_seen).sum().item()
        ce_sum += loss_output["ce_loss"].item() * batch_size
        semantic_sum += loss_output["semantic_loss"].item() * batch_size

        if max_steps and step + 1 >= max_steps:
            break

    return {
        "accuracy": correct / max(total, 1),
        "seen_accuracy": seen_correct / max(seen_total, 1),
        "unseen_accuracy": unseen_correct / max(unseen_total, 1),
        "seen_samples": seen_total,
        "unseen_samples": unseen_total,
        "ce_loss": ce_sum / max(total, 1),
        "semantic_loss": semantic_sum / max(total, 1),
    }


# 保存模型参数、结构配置、开发集指标和轮次，便于后续恢复预测。
def save_checkpoint(path, model, frame_names, args, metrics, epoch):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "frame_names": frame_names,
            "model_config": {
                "num_labels": len(frame_names),
                "dropout": args.dropout,
                "max_length": args.max_length,
            },
            "training_args": vars(args),
            "dev_metrics": metrics,
            "epoch": epoch,
        },
        path,
    )


# 程序入口：Task1：框架分类训练入口，按开发集准确率选取最佳权重。
def main():
    args = parse_args()
    if args.early_stopping_patience < 0:
        raise ValueError("early-stopping-patience must be non-negative")
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = torch.cuda.is_available() and not args.no_amp

    ensure_semantic_cache(args)
    tokenizer = BertTokenizer(vocab_file=args.vocab_file, do_lower_case=True)
    train_dataset = Task1SemanticDataset(
        args.train_file,
        args.frame_info,
        tokenizer,
        max_length=args.max_length,
    )
    dev_dataset = Task1SemanticDataset(
        args.dev_file,
        args.frame_info,
        tokenizer,
        max_length=args.max_length,
    )
    if train_dataset.frame_names != dev_dataset.frame_names:
        raise ValueError("Train and dev frame mappings differ")

    collator = Task1SemanticCollator(tokenizer.pad_token_id)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=torch.cuda.is_available(),
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=torch.cuda.is_available(),
    )

    config = BertConfig.from_json_file(args.config_file)
    model = TargetAwareFrameClassifier(
        config,
        num_labels=len(train_dataset.frame_names),
        dropout=args.dropout,
    )
    load_bert_checkpoint(model.bert, args.init_checkpoint)
    if args.gradient_checkpointing:
        model.bert.gradient_checkpointing_enable()
    model.to(device)

    semantic_distance = load_semantic_distance(
        args.semantic_cache,
        train_dataset.frame_names,
    )
    class_weights = None
    if args.class_weighting == "sqrt_inv":
        class_weights = build_class_weights(train_dataset)
    criterion = SemanticAwareFrameLoss(
        semantic_distance=semantic_distance,
        semantic_weight=args.semantic_weight,
        distance_power=args.distance_power,
        label_smoothing=args.label_smoothing,
        class_weights=class_weights,
    ).to(device)

    no_decay = ("bias", "LayerNorm.weight")
    bert_decay = []
    bert_no_decay = []
    head_decay = []
    head_no_decay = []
    for name, parameter in model.named_parameters():
        is_bert = name.startswith("bert.")
        has_no_decay = any(term in name for term in no_decay)
        if is_bert and has_no_decay:
            bert_no_decay.append(parameter)
        elif is_bert:
            bert_decay.append(parameter)
        elif has_no_decay:
            head_no_decay.append(parameter)
        else:
            head_decay.append(parameter)

    optimizer = torch.optim.AdamW(
        [
            {"params": bert_decay, "lr": args.learning_rate, "weight_decay": args.weight_decay},
            {"params": bert_no_decay, "lr": args.learning_rate, "weight_decay": 0.0},
            {"params": head_decay, "lr": args.head_learning_rate, "weight_decay": args.weight_decay},
            {"params": head_no_decay, "lr": args.head_learning_rate, "weight_decay": 0.0},
        ]
    )

    updates_per_epoch = math.ceil(
        len(train_loader) / max(args.gradient_accumulation, 1)
    )
    total_updates = args.epochs * updates_per_epoch
    if args.max_train_steps:
        total_updates = min(total_updates, args.max_train_steps)
    warmup_updates = int(total_updates * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_updates,
        num_training_steps=max(total_updates, 1),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    seen_frame_ids = collect_seen_frame_ids(train_dataset)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "run_config.json", "w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, ensure_ascii=False, indent=2)

    best_accuracy = -1.0
    epochs_without_improvement = 0
    completed_updates = 0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        # 按当前轮次调节语义损失权重；预热阶段先让分类器学习基础区分能力。
        current_semantic_weight = semantic_weight_for_epoch(
            epoch - 1,
            args.semantic_weight,
            args.semantic_warmup_epochs,
        )
        running_loss = 0.0
        running_ce = 0.0
        running_semantic = 0.0
        step_count = 0
        progress = tqdm(train_loader, desc=f"epoch {epoch}")

        for step, raw_batch in enumerate(progress):
            batch = move_batch(raw_batch, device)
            with torch.cuda.amp.autocast(enabled=use_amp):
                output = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    target_mask=batch["target_mask"],
                    content_mask=batch["content_mask"],
                )
                loss_output = criterion(
                    output["logits"],
                    batch["labels"],
                    semantic_weight=current_semantic_weight,
                )
                loss = loss_output["loss"] / args.gradient_accumulation

            # 混合精度下先缩放损失再反向传播；到累积步数后才统一更新参数。
            scaler.scale(loss).backward()
            should_update = (
                (step + 1) % args.gradient_accumulation == 0
                or step + 1 == len(train_loader)
            )
            if should_update:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                completed_updates += 1

            running_loss += loss_output["loss"].item()
            running_ce += loss_output["ce_loss"].item()
            running_semantic += loss_output["semantic_loss"].item()
            step_count += 1
            progress.set_postfix(
                loss=f"{running_loss / step_count:.4f}",
                sem=f"{running_semantic / step_count:.4f}",
                sem_w=f"{current_semantic_weight:.3f}",
            )

            if args.max_train_steps and completed_updates >= args.max_train_steps:
                break

        metrics = evaluate(
            model,
            criterion,
            dev_loader,
            device,
            seen_frame_ids,
            max_steps=args.max_eval_steps,
        )
        metrics["epoch"] = epoch
        metrics["train_loss"] = running_loss / max(step_count, 1)
        metrics["train_ce_loss"] = running_ce / max(step_count, 1)
        metrics["train_semantic_loss"] = running_semantic / max(step_count, 1)
        metrics["semantic_weight"] = current_semantic_weight
        print(json.dumps(metrics, ensure_ascii=False, indent=2))

        save_checkpoint(
            output_dir / "last.pt",
            model,
            train_dataset.frame_names,
            args,
            metrics,
            epoch,
        )
        # Task1 按开发集准确率更新 best.pt，并重置早停计数。
        if metrics["accuracy"] > best_accuracy:
            best_accuracy = metrics["accuracy"]
            epochs_without_improvement = 0
            save_checkpoint(
                output_dir / "best.pt",
                model,
                train_dataset.frame_names,
                args,
                metrics,
                epoch,
            )
            print(f"Saved new best checkpoint: accuracy={best_accuracy:.6f}")
        else:
            epochs_without_improvement += 1
            if args.early_stopping_patience > 0:
                print(
                    "Task1 early stopping: "
                    f"{epochs_without_improvement}/{args.early_stopping_patience} "
                    "epochs without dev accuracy improvement"
                )

        if args.max_train_steps and completed_updates >= args.max_train_steps:
            break
        if (
            args.early_stopping_patience > 0
            and epochs_without_improvement >= args.early_stopping_patience
        ):
            print(
                f"Task1 early stopping triggered at epoch {epoch}; "
                f"best dev accuracy={best_accuracy:.6f}"
            )
            break

    print(f"Training complete. Best dev accuracy: {best_accuracy:.6f}")


if __name__ == "__main__":
    main()
