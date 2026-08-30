#!/usr/bin/env python3
# Task2：训练跨度模型，在开发集搜索阈值，并按跨度 F1 保存最佳模型。
# 包含 FGM 对抗训练及 EMA 参数平均；该文件有独立实现，不依赖 training_utils。
# 降低预测阈值的 -0.2 偏移是在预测入口传入，不能与开发集搜索所得阈值混为一谈。
import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertConfig, BertTokenizer, get_linear_schedule_with_warmup

from build_task1_frame_semantics import load_bert_checkpoint
from model_task2_best import Task2RecallBalancedGlobalPointer
from task2_best_data import Task2BestCollator, Task2BestDataset


# 解析数据路径、模型配置及运行参数；命令行传入值会覆盖这里的默认值。
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", default="./dataset/cfn-train.json")
    parser.add_argument("--dev-file", default="./dataset/cfn-dev.json")
    parser.add_argument("--config-file", default="/root/autodl-tmp/chinese_macbert_large/config.json")
    parser.add_argument("--vocab-file", default="/root/autodl-tmp/chinese_macbert_large/vocab.txt")
    parser.add_argument(
        "--init-checkpoint", default="/root/autodl-tmp/chinese_macbert_large/pytorch_model.bin"
    )
    parser.add_argument("--output-dir", default="./saves/task2_best_macbert_large")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=0,
        help="Stop after this many consecutive epochs without dev F1 improvement; 0 disables it",
    )
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--head-learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=320)
    parser.add_argument("--inner-dim", type=int, default=64)
    parser.add_argument("--max-relative-position", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--boundary-loss-weight", type=float, default=0.2)
    parser.add_argument(
        "--hard-word-boundaries",
        action="store_true",
        help=(
            "Ablation: disable learnable soft word-boundary biases and restore "
            "word starts/ends as hard span constraints"
        ),
    )
    parser.add_argument("--fgm-epsilon", type=float, default=1.0)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--threshold-min", type=float, default=-1.0)
    parser.add_argument("--threshold-max", type=float, default=1.5)
    parser.add_argument("--threshold-step", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-steps", type=int, default=0)
    parser.add_argument("--max-eval-steps", type=int, default=0)
    parser.add_argument("--no-fgm", action="store_true")
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
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


# 将批次字段按模型接口传入，统一训练和评估时的调用方式。
def model_forward(model, batch):
    return model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        target_mask=batch["target_mask"],
        content_mask=batch["content_mask"],
        word_start_mask=batch["word_start_mask"],
        word_end_mask=batch["word_end_mask"],
        labels=batch.get("labels"),
        start_labels=batch.get("start_labels"),
        end_labels=batch.get("end_labels"),
    )


# 沿词嵌入梯度方向添加小扰动，额外反向传播后恢复原始参数。
class FGM:
    def __init__(self, model, epsilon=1.0):
        self.model = model
        self.epsilon = epsilon
        self.backup = {}

    # 备份词嵌入参数并沿归一化梯度方向添加扰动。
    def attack(self):
        self.backup = {}
        target_name = "bert.embeddings.word_embeddings"
        for name, parameter in self.model.named_parameters():
            if target_name not in name or not parameter.requires_grad:
                continue
            if parameter.grad is None:
                continue
            norm = torch.norm(parameter.grad)
            if not torch.isfinite(norm) or norm.item() == 0.0:
                continue
            self.backup[name] = parameter.data.clone()
            parameter.data.add_(self.epsilon * parameter.grad / norm)

    # 恢复扰动前备份的词嵌入参数。
    def restore(self):
        for name, parameter in self.model.named_parameters():
            if name in self.backup:
                parameter.data.copy_(self.backup[name])
        self.backup = {}


# 维护可训练参数的指数滑动平均副本，供评估和模型保存使用。
class ExponentialMovingAverage:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.num_updates = 0
        self.shadow = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        self.backup = {}

    # 用当前训练参数更新滑动平均副本，训练初期适当减小平滑系数。
    @torch.no_grad()
    def update(self, model):
        self.num_updates += 1
        # Warm up EMA so early checkpoints are not dominated by the initial
        # random task head; asymptotically this reaches the requested decay.
        decay = min(
            self.decay,
            (1.0 + self.num_updates) / (10.0 + self.num_updates),
        )
        for name, parameter in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].mul_(decay).add_(
                    parameter.detach(), alpha=1.0 - decay
                )

    # 备份当前参数并临时替换为滑动平均参数，供开发集评估与保存。
    @torch.no_grad()
    def apply(self, model):
        self.backup = {}
        for name, parameter in model.named_parameters():
            if name in self.shadow:
                self.backup[name] = parameter.detach().clone()
                parameter.data.copy_(self.shadow[name])

    # 恢复评估前备份的训练参数，并清空临时备份。
    @torch.no_grad()
    def restore(self, model):
        for name, parameter in model.named_parameters():
            if name in self.backup:
                parameter.data.copy_(self.backup[name])
        self.backup = {}


# 按照给定范围及步长构造待评估阈值，检查范围是否有效。
def build_thresholds(minimum, maximum, step):
    if step <= 0 or maximum < minimum:
        raise ValueError("Invalid threshold range")
    count = int(round((maximum - minimum) / step))
    return [round(minimum + index * step, 6) for index in range(count + 1)]


# 逐个阈值统计跨度精确率、召回率和 F1，返回 F1 最佳阈值及完整搜索结果。
@torch.no_grad()
def evaluate(model, loader, device, thresholds, max_steps=0):
    model.eval()
    counts = {
        threshold: {"tp": 0, "pred": 0, "gold": 0}
        for threshold in thresholds
    }
    loss_sums = {"eval_loss": 0.0, "span_loss": 0.0, "boundary_loss": 0.0}
    samples = 0
    for step, raw_batch in enumerate(tqdm(loader, desc="eval task2 best")):
        batch = move_batch(raw_batch, device)
        output = model_forward(model, batch)
        gold = batch["labels"].bool()
        batch_size = gold.shape[0]
        samples += batch_size
        loss_sums["eval_loss"] += output["loss"].item() * batch_size
        loss_sums["span_loss"] += output["span_loss"].item() * batch_size
        loss_sums["boundary_loss"] += output["boundary_loss"].item() * batch_size
        gold_count = gold.sum().item()
        for threshold in thresholds:
            predicted = (output["logits"] >= threshold) & output["span_mask"]
            counts[threshold]["tp"] += (predicted & gold).sum().item()
            counts[threshold]["pred"] += predicted.sum().item()
            counts[threshold]["gold"] += gold_count
        if max_steps and step + 1 >= max_steps:
            break

    rows = []
    for threshold in thresholds:
        count = counts[threshold]
        precision = count["tp"] / max(count["pred"], 1)
        recall = count["tp"] / max(count["gold"], 1)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
        rows.append(
            {
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "avg_predicted_spans": count["pred"] / max(samples, 1),
                "avg_gold_spans": count["gold"] / max(samples, 1),
                **count,
            }
        )
    # 在当前模型的开发集阈值搜索结果中选择 F1 最高的阈值。
    best = dict(max(rows, key=lambda row: row["f1"]))
    for name, value in loss_sums.items():
        best[name] = value / max(samples, 1)
    best["threshold_results"] = rows
    return best


# 保存模型参数、结构配置、开发集指标和轮次，便于后续恢复预测。
def save_checkpoint(path, model, args, metrics, epoch):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": {
                "inner_dim": args.inner_dim,
                "max_relative_position": args.max_relative_position,
                "max_length": args.max_length,
                "dropout": args.dropout,
                "boundary_loss_weight": args.boundary_loss_weight,
                "soft_word_boundaries": not args.hard_word_boundaries,
                "threshold": metrics["threshold"],
            },
            "training_args": vars(args),
            "dev_metrics": metrics,
            "epoch": epoch,
        },
        path,
    )


# 程序入口：Task2：训练跨度模型，在开发集搜索阈值，并按跨度 F1 保存最佳模型。
def main():
    args = parse_args()
    if args.early_stopping_patience < 0:
        raise ValueError("early-stopping-patience must be non-negative")
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = torch.cuda.is_available() and not args.no_amp
    thresholds = build_thresholds(
        args.threshold_min, args.threshold_max, args.threshold_step
    )

    tokenizer = BertTokenizer(vocab_file=args.vocab_file, do_lower_case=True)
    train_dataset = Task2BestDataset(args.train_file, tokenizer, args.max_length)
    dev_dataset = Task2BestDataset(args.dev_file, tokenizer, args.max_length)
    collator = Task2BestCollator(tokenizer.pad_token_id)
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
    model = Task2RecallBalancedGlobalPointer(
        config,
        inner_dim=args.inner_dim,
        max_relative_position=args.max_relative_position,
        dropout=args.dropout,
        boundary_loss_weight=args.boundary_loss_weight,
        soft_word_boundaries=not args.hard_word_boundaries,
    )
    load_bert_checkpoint(model.bert, args.init_checkpoint)
    if args.gradient_checkpointing:
        model.bert.gradient_checkpointing_enable()
    model.to(device)

    no_decay = ("bias", "LayerNorm.weight")
    parameter_groups = {
        "bert_decay": [],
        "bert_no_decay": [],
        "head_decay": [],
        "head_no_decay": [],
    }
    for name, parameter in model.named_parameters():
        prefix = "bert" if name.startswith("bert.") else "head"
        suffix = "no_decay" if any(term in name for term in no_decay) else "decay"
        parameter_groups[f"{prefix}_{suffix}"].append(parameter)
    optimizer = torch.optim.AdamW(
        [
            {
                "params": parameter_groups["bert_decay"],
                "lr": args.learning_rate,
                "weight_decay": args.weight_decay,
            },
            {
                "params": parameter_groups["bert_no_decay"],
                "lr": args.learning_rate,
                "weight_decay": 0.0,
            },
            {
                "params": parameter_groups["head_decay"],
                "lr": args.head_learning_rate,
                "weight_decay": args.weight_decay,
            },
            {
                "params": parameter_groups["head_no_decay"],
                "lr": args.head_learning_rate,
                "weight_decay": 0.0,
            },
        ]
    )

    updates_per_epoch = math.ceil(
        len(train_loader) / args.gradient_accumulation
    )
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
                "device": str(device),
                "train_samples": len(train_dataset),
                "dev_samples": len(dev_dataset),
                "epochs": args.epochs,
                "fgm": fgm is not None,
                "ema": ema is not None,
                "thresholds": thresholds,
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    best_f1 = -1.0
    epochs_without_improvement = 0
    completed_updates = 0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = {"loss": 0.0, "span_loss": 0.0, "boundary_loss": 0.0}
        steps = 0
        progress = tqdm(train_loader, desc=f"task2 best epoch {epoch}")
        for step, raw_batch in enumerate(progress):
            batch = move_batch(raw_batch, device)
            with torch.cuda.amp.autocast(enabled=use_amp):
                output = model_forward(model, batch)
                scaled_loss = output["loss"] / args.gradient_accumulation
            scaler.scale(scaled_loss).backward()

            if fgm is not None:
                # 临时扰动词嵌入，在扰动输入下额外计算损失并累积梯度。
                fgm.attack()
                with torch.cuda.amp.autocast(enabled=use_amp):
                    adversarial_output = model_forward(model, batch)
                    adversarial_loss = (
                        adversarial_output["loss"] / args.gradient_accumulation
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

            running["loss"] += output["loss"].item()
            running["span_loss"] += output["span_loss"].item()
            running["boundary_loss"] += output["boundary_loss"].item()
            steps += 1
            progress.set_postfix(
                loss=f"{running['loss'] / max(steps, 1):.4f}"
            )
            if args.max_train_steps and completed_updates >= args.max_train_steps:
                break

        if ema is not None:
            # 临时使用滑动平均参数评估；若本轮保存 checkpoint，保存的也是该组参数。
            ema.apply(model)
        metrics = evaluate(
            model, dev_loader, device, thresholds, args.max_eval_steps
        )
        metrics["epoch"] = epoch
        metrics["train_loss"] = running["loss"] / max(steps, 1)
        metrics["train_span_loss"] = running["span_loss"] / max(steps, 1)
        metrics["train_boundary_loss"] = running["boundary_loss"] / max(steps, 1)
        print(json.dumps(metrics, ensure_ascii=False, indent=2))

        # Checkpoints contain the same EMA parameters used for dev evaluation.
        save_checkpoint(output_dir / "last.pt", model, args, metrics, epoch)
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            epochs_without_improvement = 0
            save_checkpoint(output_dir / "best.pt", model, args, metrics, epoch)
            print(f"Saved new best Task 2 checkpoint: F1={best_f1:.6f}")
        else:
            epochs_without_improvement += 1
            if args.early_stopping_patience > 0:
                print(
                    "Task2 early stopping: "
                    f"{epochs_without_improvement}/{args.early_stopping_patience} "
                    "epochs without dev F1 improvement"
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
                f"Task2 early stopping triggered at epoch {epoch}; "
                f"best dev F1={best_f1:.6f}"
            )
            break

    print(f"Task 2 best training complete. Best dev F1: {best_f1:.6f}")


if __name__ == "__main__":
    main()
