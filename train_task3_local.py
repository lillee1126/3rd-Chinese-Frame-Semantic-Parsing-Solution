#!/usr/bin/env python3
# Task3：角色分类训练入口，加入框架噪声、FGM 及 EMA。
# 同时评估真实上游标注条件（oracle）和上游预测条件（pipeline）；按 pipeline F1 选择最佳权重。
# 开发集的 Task1/Task2 预测文件须事先生成；训练中的框架噪声不用于正常预测。
import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertConfig, BertTokenizer, get_linear_schedule_with_warmup

from build_task1_frame_semantics import load_bert_checkpoint
from model_task3_local import FrameLocalRoleClassifier
from task3_local_data import (
    Task3LocalCollator,
    Task3LocalDataset,
    Task3PipelineDevDataset,
)
from training_utils import ExponentialMovingAverage, FGM, move_batch, set_seed


# 解析数据路径、模型配置及运行参数；命令行传入值会覆盖这里的默认值。
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", default="./dataset/cfn-train.json")
    parser.add_argument("--dev-file", default="./dataset/cfn-dev.json")
    parser.add_argument("--frame-info", default="./dataset/frame_info.json")
    parser.add_argument("--pipeline-dev-task1", required=True)
    parser.add_argument("--pipeline-dev-task2", required=True)
    parser.add_argument("--config-file", default="/root/autodl-tmp/chinese_macbert_large/config.json")
    parser.add_argument("--vocab-file", default="/root/autodl-tmp/chinese_macbert_large/vocab.txt")
    parser.add_argument(
        "--init-checkpoint", default="/root/autodl-tmp/chinese_macbert_large/pytorch_model.bin"
    )
    parser.add_argument(
        "--semantic-cache", default="./knowledge/task3_local_semantics_macbert_large.pt"
    )
    parser.add_argument("--output-dir", default="./saves/task3_local_macbert_large")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--semantic-cache-batch-size", type=int, default=48)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=0,
        help="Stop after this many consecutive epochs without pipeline dev F1 improvement; 0 disables it",
    )
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--head-learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=320)
    parser.add_argument("--relative-position-dim", type=int, default=32)
    parser.add_argument("--max-relative-position", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--frame-corruption-rate", type=float, default=0.05)
    parser.add_argument("--frame-dropout-rate", type=float, default=0.05)
    parser.add_argument("--initial-local-scale", type=float, default=5.0)
    parser.add_argument("--initial-outside-penalty", type=float, default=4.0)
    parser.add_argument("--fgm-epsilon", type=float, default=1.0)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-steps", type=int, default=0)
    parser.add_argument("--max-eval-steps", type=int, default=0)
    parser.add_argument("--rebuild-semantic-cache", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--no-fgm", action="store_true")
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


# 缓存不存在或要求重建时，调用配套语义构建脚本生成缓存。
def ensure_semantic_cache(args):
    path = Path(args.semantic_cache)
    if path.exists() and not args.rebuild_semantic_cache:
        return
    command = [
        sys.executable,
        str(Path(__file__).with_name("build_task3_local_semantics.py")),
        "--frame-info",
        args.frame_info,
        "--config-file",
        args.config_file,
        "--vocab-file",
        args.vocab_file,
        "--init-checkpoint",
        args.init_checkpoint,
        "--output",
        args.semantic_cache,
        "--batch-size",
        str(args.semantic_cache_batch_size),
    ]
    print("Task3 local semantic cache is missing; building it now.")
    subprocess.run(command, check=True)


# 将批次字段按模型接口传入，统一训练和评估时的调用方式。
def model_forward(model, batch, frame_ids=None, frame_strength=None):
    return model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        target_mask=batch["target_mask"],
        frame_ids=batch["frame_ids"] if frame_ids is None else frame_ids,
        span_starts=batch["span_starts"],
        span_ends=batch["span_ends"],
        span_mask=batch["span_mask"],
        role_targets=batch.get("role_targets"),
        frame_strength=frame_strength,
    )


# 随机替换部分框架ID，另将部分样本的框架强度置零，模拟上游错误或信息缺失。
def noisy_frame_inputs(frame_ids, num_frames, corruption_rate, dropout_rate):
    draw = torch.rand(frame_ids.shape, device=frame_ids.device)
    corrupt = draw < corruption_rate
    drop = (draw >= corruption_rate) & (
        draw < corruption_rate + dropout_rate
    )
    offsets = torch.randint(1, num_frames, frame_ids.shape, device=frame_ids.device)
    corrupted_ids = torch.where(corrupt, (frame_ids + offsets) % num_frames, frame_ids)
    strength = torch.ones_like(frame_ids, dtype=torch.float)
    strength = strength.masked_fill(drop, 0.0)
    return corrupted_ids, strength


# 每个有效跨度选择最高分角色；流水线评估用全体真实角色数计算召回率，计入上游漏检。
@torch.no_grad()
def evaluate(model, loader, device, gold_role_count=None, max_steps=0, desc="eval"):
    model.eval()
    true_positive = 0
    predicted_count = 0
    observed_gold_count = 0
    loss_sum = 0.0
    span_total = 0
    for step, raw_batch in enumerate(tqdm(loader, desc=desc)):
        batch = move_batch(raw_batch, device)
        output = model_forward(model, batch)
        predictions = output["logits"].argmax(dim=-1)
        predicted_is_gold = torch.gather(
            batch["role_targets"],
            dim=-1,
            index=predictions.unsqueeze(-1),
        ).squeeze(-1).bool()
        valid = batch["span_mask"]
        true_positive += (predicted_is_gold & valid).sum().item()
        predicted_count += valid.sum().item()
        observed_gold_count += (
            batch["role_targets"].bool() & valid.unsqueeze(-1)
        ).sum().item()
        span_count = valid.sum().item()
        span_total += span_count
        if output["loss"] is not None:
            loss_sum += output["loss"].item() * span_count
        if max_steps and step + 1 >= max_steps:
            break
    final_gold_count = (
        observed_gold_count if gold_role_count is None else gold_role_count
    )
    precision = true_positive / max(predicted_count, 1)
    recall = true_positive / max(final_gold_count, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": true_positive,
        "pred": predicted_count,
        "gold": final_gold_count,
        "eval_loss": loss_sum / max(span_total, 1),
    }


# 保存模型参数、结构配置、开发集指标和轮次，便于后续恢复预测。
def save_checkpoint(path, model, dataset, args, oracle, pipeline, epoch):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "frame_names": dataset.frame_names,
            "role_names": dataset.role_names,
            "model_config": {
                "max_length": args.max_length,
                "relative_position_dim": args.relative_position_dim,
                "max_relative_position": args.max_relative_position,
                "dropout": args.dropout,
                "initial_local_scale": args.initial_local_scale,
                "initial_outside_penalty": args.initial_outside_penalty,
            },
            "training_args": vars(args),
            "oracle_dev_metrics": oracle,
            "pipeline_dev_metrics": pipeline,
            "epoch": epoch,
        },
        path,
    )


# 程序入口：Task3：角色分类训练入口，加入框架噪声、FGM 及 EMA。
def main():
    args = parse_args()
    if args.early_stopping_patience < 0:
        raise ValueError("early-stopping-patience must be non-negative")
    if args.frame_corruption_rate + args.frame_dropout_rate >= 1.0:
        raise ValueError("Frame corruption + dropout rate must be below 1")
    set_seed(args.seed)
    ensure_semantic_cache(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = torch.cuda.is_available() and not args.no_amp
    tokenizer = BertTokenizer(vocab_file=args.vocab_file, do_lower_case=True)
    train_dataset = Task3LocalDataset(
        args.train_file, args.frame_info, tokenizer, args.max_length
    )
    oracle_dev_dataset = Task3LocalDataset(
        args.dev_file, args.frame_info, tokenizer, args.max_length
    )
    pipeline_dev_dataset = Task3PipelineDevDataset(
        args.dev_file,
        args.frame_info,
        tokenizer,
        args.pipeline_dev_task1,
        args.pipeline_dev_task2,
        args.max_length,
    )
    semantics = torch.load(args.semantic_cache, map_location="cpu")
    if semantics["frame_names"] != train_dataset.frame_names:
        raise ValueError("Local semantic cache frame order differs from frame_info")
    if semantics["role_names"] != train_dataset.role_names:
        raise ValueError("Local semantic cache role order differs from frame_info")

    collator = Task3LocalCollator(
        len(train_dataset.role_names), tokenizer.pad_token_id
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=torch.cuda.is_available(),
    )
    oracle_dev_loader = DataLoader(
        oracle_dev_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=torch.cuda.is_available(),
    )
    pipeline_dev_loader = DataLoader(
        pipeline_dev_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=torch.cuda.is_available(),
    )

    config = BertConfig.from_json_file(args.config_file)
    model = FrameLocalRoleClassifier(
        config,
        semantics["frame_embeddings"],
        semantics["frame_role_embeddings"],
        semantics["frame_role_ids"],
        len(train_dataset.role_names),
        relative_position_dim=args.relative_position_dim,
        max_relative_position=args.max_relative_position,
        dropout=args.dropout,
        initial_local_scale=args.initial_local_scale,
        initial_outside_penalty=args.initial_outside_penalty,
    )
    load_bert_checkpoint(model.bert, args.init_checkpoint)
    if args.gradient_checkpointing:
        model.bert.gradient_checkpointing_enable()
    model.to(device)

    no_decay = ("bias", "LayerNorm.weight")
    groups = {"bert_decay": [], "bert_no_decay": [], "head_decay": [], "head_no_decay": []}
    for name, parameter in model.named_parameters():
        prefix = "bert" if name.startswith("bert.") else "head"
        suffix = "no_decay" if any(term in name for term in no_decay) else "decay"
        groups[f"{prefix}_{suffix}"].append(parameter)
    optimizer = torch.optim.AdamW(
        [
            {"params": groups["bert_decay"], "lr": args.learning_rate, "weight_decay": args.weight_decay},
            {"params": groups["bert_no_decay"], "lr": args.learning_rate, "weight_decay": 0.0},
            {"params": groups["head_decay"], "lr": args.head_learning_rate, "weight_decay": args.weight_decay},
            {"params": groups["head_no_decay"], "lr": args.head_learning_rate, "weight_decay": 0.0},
        ]
    )
    updates_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation)
    total_updates = args.epochs * updates_per_epoch
    if args.max_train_steps:
        total_updates = min(total_updates, args.max_train_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        int(total_updates * args.warmup_ratio),
        max(total_updates, 1),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    fgm = None if args.no_fgm else FGM(model, args.fgm_epsilon)
    ema = None if args.no_ema else ExponentialMovingAverage(model, args.ema_decay)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "run_config.json", "w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, ensure_ascii=False, indent=2)

    print(
        json.dumps(
            {
                "train_samples": len(train_dataset),
                "oracle_dev_samples": len(oracle_dev_dataset),
                "pipeline_dev_samples": len(pipeline_dev_dataset),
                "pipeline_dev_gold_roles": pipeline_dev_dataset.pipeline_gold_role_count,
                "frame_role_occurrences": int(semantics["role_occurrences"]),
                "maximum_roles_per_frame": int(semantics["maximum_roles"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    best_pipeline_f1 = -1.0
    epochs_without_improvement = 0
    completed_updates = 0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        steps = 0
        progress = tqdm(train_loader, desc=f"task3 local epoch {epoch}")
        for step, raw_batch in enumerate(progress):
            batch = move_batch(raw_batch, device)
            # 仅训练时随机替换/丢弃部分框架信息，提高对 Task1 错误的容忍度。
            noisy_frames, frame_strength = noisy_frame_inputs(
                batch["frame_ids"],
                model.num_frames,
                args.frame_corruption_rate,
                args.frame_dropout_rate,
            )
            with torch.cuda.amp.autocast(enabled=use_amp):
                output = model_forward(
                    model, batch, noisy_frames, frame_strength
                )
                loss = output["loss"] / args.gradient_accumulation
            # 混合精度下先缩放损失再反向传播；到累积步数后才统一更新参数。
            scaler.scale(loss).backward()
            if fgm is not None:
                # 临时扰动词嵌入，在扰动输入下额外计算损失并累积梯度。
                fgm.attack()
                with torch.cuda.amp.autocast(enabled=use_amp):
                    adversarial = model_forward(
                        model, batch, noisy_frames, frame_strength
                    )
                    adversarial_loss = (
                        adversarial["loss"] / args.gradient_accumulation
                    )
                scaler.scale(adversarial_loss).backward()
                # 对抗梯度已保留，但参数必须恢复后再做正常优化更新。
                fgm.restore()

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
                if ema is not None:
                    ema.update(model)
            running_loss += output["loss"].item()
            steps += 1
            progress.set_postfix(loss=f"{running_loss / steps:.4f}")
            if args.max_train_steps and completed_updates >= args.max_train_steps:
                break

        if ema is not None:
            # 临时使用滑动平均参数评估；若本轮保存 checkpoint，保存的也是该组参数。
            ema.apply(model)
        oracle = evaluate(
            model,
            oracle_dev_loader,
            device,
            max_steps=args.max_eval_steps,
            desc="oracle dev task3 local",
        )
        pipeline = evaluate(
            model,
            pipeline_dev_loader,
            device,
            gold_role_count=pipeline_dev_dataset.pipeline_gold_role_count,
            max_steps=args.max_eval_steps,
            desc="pipeline dev task3 local",
        )
        metrics = {
            "epoch": epoch,
            "train_loss": running_loss / max(steps, 1),
            "oracle_dev": oracle,
            "pipeline_dev": pipeline,
            "outside_penalty": float(
                torch.nn.functional.softplus(model.raw_outside_penalty.detach())
            ),
            "local_scale": float(
                model.log_local_scale.detach().exp().clamp(max=20.0)
            ),
        }
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        save_checkpoint(
            output_dir / "last.pt", model, train_dataset, args, oracle, pipeline, epoch
        )
        # 依据真实上游预测条件下的 F1 选模，而不是依据理想标注条件下的分数。
        if pipeline["f1"] > best_pipeline_f1:
            best_pipeline_f1 = pipeline["f1"]
            epochs_without_improvement = 0
            save_checkpoint(
                output_dir / "best.pt", model, train_dataset, args, oracle, pipeline, epoch
            )
            print(
                f"Saved new best Task3 local checkpoint: "
                f"pipeline_F1={best_pipeline_f1:.6f}"
            )
        else:
            epochs_without_improvement += 1
            if args.early_stopping_patience > 0:
                print(
                    "Task3 early stopping: "
                    f"{epochs_without_improvement}/{args.early_stopping_patience} "
                    "epochs without pipeline dev F1 improvement"
                )
        if ema is not None:
            # 评估和保存结束，恢复原训练参数以继续下一轮优化。
            ema.restore(model)
        if args.max_train_steps and completed_updates >= args.max_train_steps:
            break
        if (
            args.early_stopping_patience > 0
            and epochs_without_improvement >= args.early_stopping_patience
        ):
            print(
                f"Task3 early stopping triggered at epoch {epoch}; "
                f"best pipeline dev F1={best_pipeline_f1:.6f}"
            )
            break
    print(
        f"Task3 local training complete. Best pipeline dev F1: "
        f"{best_pipeline_f1:.6f}"
    )


if __name__ == "__main__":
    main()
