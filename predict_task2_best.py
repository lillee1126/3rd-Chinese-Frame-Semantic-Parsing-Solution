#!/usr/bin/env python3
# Task2：读取 checkpoint 和阈值配置，预测论元起止坐标。
# 输出 JSON 的每行为 [句子ID, 原句起点, 原句终点]，坐标为从 0 开始的闭区间。
# 最终方案由外部脚本传入 --threshold-offset -0.2；本文件的偏移默认值仍是 0。
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertConfig, BertTokenizer

from model_task2_best import Task2RecallBalancedGlobalPointer
from task2_best_data import Task2BestCollator, Task2BestDataset


# 解析数据路径、模型配置及运行参数；命令行传入值会覆盖这里的默认值。
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-file", default="./dataset/cfn-test-B.json")
    parser.add_argument("--config-file", default="/root/autodl-tmp/chinese_macbert_large/config.json")
    parser.add_argument("--vocab-file", default="/root/autodl-tmp/chinese_macbert_large/vocab.txt")
    parser.add_argument("--checkpoint", default="./saves/task2_best_macbert_large/best.pt")
    parser.add_argument("--output", default="./dataset/B_task2_best_test.json")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument(
        "--threshold-offset",
        type=float,
        default=0.0,
        help="Add this value to the checkpoint threshold; a negative value raises recall",
    )
    parser.add_argument(
        "--recall-f1-tolerance",
        type=float,
        default=0.0,
        help=(
            "Among dev thresholds within this absolute F1 distance of the best, "
            "select the one with highest recall before applying threshold-offset"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


# 程序入口：Task2：读取 checkpoint 和阈值配置，预测论元起止坐标。
@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model_config = checkpoint["model_config"]
    checkpoint_threshold = (
        float(model_config["threshold"])
        if args.threshold is None
        else args.threshold
    )
    calibrated_threshold = checkpoint_threshold
    calibration_row = None
    threshold_rows = checkpoint.get("dev_metrics", {}).get(
        "threshold_results", []
    )
    if args.threshold is None and args.recall_f1_tolerance > 0 and threshold_rows:
        best_dev_f1 = max(float(row["f1"]) for row in threshold_rows)
        eligible = [
            row
            for row in threshold_rows
            if float(row["f1"]) >= best_dev_f1 - args.recall_f1_tolerance
        ]
        calibration_row = max(
            eligible,
            key=lambda row: (float(row["recall"]), float(row["f1"])),
        )
        calibrated_threshold = float(calibration_row["threshold"])
    # 有效阈值 = 选定阈值 + 偏移；主方案传入 -0.2，扩大候选集合以提高召回。
    threshold = calibrated_threshold + args.threshold_offset

    tokenizer = BertTokenizer(vocab_file=args.vocab_file, do_lower_case=True)
    dataset = Task2BestDataset(
        args.test_file,
        tokenizer,
        max_length=int(model_config["max_length"]),
        for_test=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=Task2BestCollator(tokenizer.pad_token_id),
        pin_memory=torch.cuda.is_available(),
    )

    config = BertConfig.from_json_file(args.config_file)
    model = Task2RecallBalancedGlobalPointer(
        config,
        inner_dim=int(model_config["inner_dim"]),
        max_relative_position=int(model_config["max_relative_position"]),
        dropout=float(model_config["dropout"]),
        boundary_loss_weight=float(model_config["boundary_loss_weight"]),
        soft_word_boundaries=bool(
            model_config.get("soft_word_boundaries", True)
        ),
    )
    # 严格核对模型结构和参数键名，不允许静默遗漏权重。
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()

    predictions = []
    for raw_batch in tqdm(loader, desc="predict task2 best"):
        batch = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in raw_batch.items()
        }
        output = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            target_mask=batch["target_mask"],
            content_mask=batch["content_mask"],
            word_start_mask=batch["word_start_mask"],
            word_end_mask=batch["word_end_mask"],
        )
        predicted = (output["logits"] >= threshold) & output["span_mask"]
        for batch_index, start, end in torch.nonzero(predicted).cpu().tolist():
            token_to_char = batch["token_to_chars"][batch_index]
            if start >= len(token_to_char) or end >= len(token_to_char):
                continue
            # 从含目标标记的 token 坐标还原为原句字符坐标，终点同理。
            original_start = token_to_char[start]
            original_end = token_to_char[end]
            if 0 <= original_start <= original_end < batch["text_lengths"][batch_index]:
                predictions.append(
                    [
                        batch["sentence_ids"][batch_index],
                        original_start,
                        original_end,
                    ]
                )

    predictions = sorted(set(tuple(row) for row in predictions))
    predictions = [list(row) for row in predictions]
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="\n") as handle:
        # 保留中文标签，写出最终预测 JSON。
        json.dump(predictions, handle, ensure_ascii=False, indent=1)
    print(
        f"Saved {len(predictions)} Task 2 spans to {output_path}; "
        f"checkpoint_threshold={checkpoint_threshold}; "
        f"calibrated_threshold={calibrated_threshold}; "
        f"threshold_offset={args.threshold_offset}; "
        f"effective_threshold={threshold}; "
        f"checkpoint_epoch={checkpoint['epoch']}"
    )
    if calibration_row is not None:
        print(
            "Recall calibration dev metrics: "
            f"P={float(calibration_row['precision']):.6f}, "
            f"R={float(calibration_row['recall']):.6f}, "
            f"F1={float(calibration_row['f1']):.6f}"
        )


if __name__ == "__main__":
    main()
