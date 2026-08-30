#!/usr/bin/env python3
# Task3：读取 Task1 框架、Task2 跨度以及语义缓存，预测每个跨度的角色。
# 输出 JSON 的每行为 [句子ID, 原句起点, 原句终点, 角色名称]。
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertConfig, BertTokenizer

from model_task3_local import FrameLocalRoleClassifier
from task3_local_data import Task3LocalCollator, Task3LocalDataset


# 解析数据路径、模型配置及运行参数；命令行传入值会覆盖这里的默认值。
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-file", default="./dataset/cfn-test-B.json")
    parser.add_argument("--frame-info", default="./dataset/frame_info.json")
    parser.add_argument("--task1-file", default="./dataset/B_task1_test.json")
    parser.add_argument("--task2-file", default="./dataset/B_task2_test.json")
    parser.add_argument("--config-file", default="/root/autodl-tmp/chinese_macbert_large/config.json")
    parser.add_argument("--vocab-file", default="/root/autodl-tmp/chinese_macbert_large/vocab.txt")
    parser.add_argument(
        "--semantic-cache", default="./knowledge/task3_local_semantics_macbert_large.pt"
    )
    parser.add_argument("--checkpoint", default="./saves/task3_local_macbert_large/best.pt")
    parser.add_argument("--output", default="./dataset/B_task3_local_test.json")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


# 程序入口：Task3：读取 Task1 框架、Task2 跨度以及语义缓存，预测每个跨度的角色。
@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    semantics = torch.load(args.semantic_cache, map_location="cpu")
    model_config = checkpoint["model_config"]
    with open(args.task2_file, "r", encoding="utf-8") as handle:
        unique_task2_spans = {tuple(row) for row in json.load(handle)}

    tokenizer = BertTokenizer(vocab_file=args.vocab_file, do_lower_case=True)
    dataset = Task3LocalDataset(
        args.test_file,
        args.frame_info,
        tokenizer,
        max_length=int(model_config["max_length"]),
        task1_path=args.task1_file,
        task2_path=args.task2_file,
        for_test=True,
    )
    # 检查权重、知识缓存和数据的标签顺序，避免编号一致但语义错位。
    if checkpoint["frame_names"] != dataset.frame_names:
        raise ValueError("Checkpoint frame order differs from frame_info")
    if checkpoint["role_names"] != dataset.role_names:
        raise ValueError("Checkpoint role order differs from frame_info")
    if semantics["frame_names"] != dataset.frame_names:
        raise ValueError("Local semantics frame order differs from frame_info")
    if semantics["role_names"] != dataset.role_names:
        raise ValueError("Local semantics role order differs from frame_info")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=Task3LocalCollator(
            len(dataset.role_names), tokenizer.pad_token_id, for_test=True
        ),
        pin_memory=torch.cuda.is_available(),
    )
    config = BertConfig.from_json_file(args.config_file)
    model = FrameLocalRoleClassifier(
        config,
        semantics["frame_embeddings"],
        semantics["frame_role_embeddings"],
        semantics["frame_role_ids"],
        len(dataset.role_names),
        relative_position_dim=int(model_config["relative_position_dim"]),
        max_relative_position=int(model_config["max_relative_position"]),
        dropout=float(model_config["dropout"]),
        initial_local_scale=float(model_config["initial_local_scale"]),
        initial_outside_penalty=float(model_config["initial_outside_penalty"]),
    )
    # 严格核对模型结构和参数键名，不允许静默遗漏权重。
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()

    predictions = []
    internal_spans = 0
    for raw_batch in tqdm(loader, desc="predict task3 local"):
        batch = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in raw_batch.items()
        }
        output = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            target_mask=batch["target_mask"],
            frame_ids=batch["frame_ids"],
            span_starts=batch["span_starts"],
            span_ends=batch["span_ends"],
            span_mask=batch["span_mask"],
        )
        # 每个跨度仅选择一个最高分角色；多标签训练并不意味着输出多个角色。
        predicted_roles = output["logits"].argmax(dim=-1).cpu().tolist()
        for batch_index, spans in enumerate(batch["original_spans"]):
            sentence_id = batch["sentence_ids"][batch_index]
            for span_index, (start, end) in enumerate(spans):
                internal_spans += 1
                role_id = predicted_roles[batch_index][span_index]
                predictions.append(
                    [sentence_id, start, end, dataset.role_names[role_id]]
                )
    predictions = sorted(set(tuple(row) for row in predictions))
    predictions = [list(row) for row in predictions]
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="\n") as handle:
        # 保留中文标签，写出最终预测 JSON。
        json.dump(predictions, handle, ensure_ascii=False, indent=1)
    print(
        f"Saved {len(predictions)} Task3 local roles for "
        f"{len(unique_task2_spans)} unique Task2 spans to {output_path}; "
        f"internal_span_rows={internal_spans}; checkpoint_epoch={checkpoint['epoch']}; "
        f"pipeline_dev_f1={checkpoint['pipeline_dev_metrics']['f1']:.6f}"
    )


if __name__ == "__main__":
    main()
