#!/usr/bin/env python3
# Task1：加载训练好的 checkpoint，预测原句目标词对应的框架。
# 输出 JSON 的每行为 [句子ID, 框架名称]，供提交或 Task3 使用。
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertConfig, BertTokenizer

from model_task1_semantic import TargetAwareFrameClassifier
from task1_semantic_data import Task1SemanticCollator, Task1SemanticDataset


# 解析数据路径、模型配置及运行参数；命令行传入值会覆盖这里的默认值。
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-file", default="./dataset/cfn-test-B.json")
    parser.add_argument("--frame-info", default="./dataset/frame_info.json")
    parser.add_argument("--config-file", default="/root/autodl-tmp/chinese_macbert_large/config.json")
    parser.add_argument("--vocab-file", default="/root/autodl-tmp/chinese_macbert_large/vocab.txt")
    parser.add_argument(
        "--checkpoint",
        default="./saves/task1_semantic_p2_w1_macbert_large/best.pt",
    )
    parser.add_argument("--output", default="./dataset/B_task1_test.json")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-predict-steps", type=int, default=0)
    return parser.parse_args()


# 程序入口：Task1：加载训练好的 checkpoint，预测原句目标词对应的框架。
@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model_config = checkpoint["model_config"]

    tokenizer = BertTokenizer(vocab_file=args.vocab_file, do_lower_case=True)
    dataset = Task1SemanticDataset(
        args.test_file,
        args.frame_info,
        tokenizer,
        max_length=int(model_config["max_length"]),
        for_test=True,
    )
    if checkpoint["frame_names"] != dataset.frame_names:
        raise ValueError(
            "Checkpoint frame order differs from frame_info.json; refusing to predict."
        )

    config = BertConfig.from_json_file(args.config_file)
    model = TargetAwareFrameClassifier(
        config,
        num_labels=int(model_config["num_labels"]),
        dropout=float(model_config["dropout"]),
    )
    # 严格核对模型结构和参数键名，不允许静默遗漏权重。
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    model.eval()

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=Task1SemanticCollator(tokenizer.pad_token_id),
        pin_memory=torch.cuda.is_available(),
    )
    predictions = []
    for step, batch in enumerate(tqdm(loader, desc="predict task1")):
        tensor_batch = {
            key: value.to(device)
            for key, value in batch.items()
            if torch.is_tensor(value)
        }
        output = model(
            input_ids=tensor_batch["input_ids"],
            attention_mask=tensor_batch["attention_mask"],
            target_mask=tensor_batch["target_mask"],
            content_mask=tensor_batch["content_mask"],
        )
        predicted_ids = output["logits"].argmax(dim=-1).cpu().tolist()
        for sentence_id, frame_id in zip(batch["sentence_ids"], predicted_ids):
            predictions.append([sentence_id, dataset.frame_names[frame_id]])
        if args.max_predict_steps and step + 1 >= args.max_predict_steps:
            break

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="\n") as handle:
        # 保留中文标签，写出最终预测 JSON。
        json.dump(predictions, handle, ensure_ascii=False, indent=1)
    print(f"Saved {len(predictions)} Task 1 predictions to {output_path}")


if __name__ == "__main__":
    main()
