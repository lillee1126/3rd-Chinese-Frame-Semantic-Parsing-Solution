#!/usr/bin/env python3
# Task1：将框架名称和定义编码为向量，构建框架间语义距离缓存。
# 缓存供训练中的语义损失使用；本文件还提供其他任务复用的权重加载函数。
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import BertConfig, BertModel, BertTokenizer


# 从可能带有外层包装的 checkpoint 中提取实际参数字典。
def unwrap_state_dict(value):
    if isinstance(value, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            nested = value.get(key)
            if isinstance(nested, dict):
                return nested
    return value


# 适配预训练权重键名并加载编码器参数，供三个任务复用。
def load_bert_checkpoint(bert, checkpoint_path):
    state = unwrap_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    bert_state = {}
    for key, value in state.items():
        if key.startswith("bert."):
            bert_state[key[len("bert."):]] = value
        elif not key.startswith("cls."):
            bert_state[key] = value
    incompatible = bert.load_state_dict(bert_state, strict=False)
    if incompatible.missing_keys:
        print(f"Warning: missing BERT keys: {len(incompatible.missing_keys)}")
    if incompatible.unexpected_keys:
        print(f"Warning: unexpected BERT keys: {len(incompatible.unexpected_keys)}")


# 将框架名称及定义整理成供预训练编码器读取的文本。
def build_frame_text(frame):
    name = str(frame["frame_name"]).strip()
    definition = str(frame.get("frame_def", "")).strip()
    if definition:
        return f"框架名称：{name}。框架定义：{definition}"
    return f"框架名称：{name}。"


# 分批编码文本，对有效文本 token 做平均池化，返回语义向量。
@torch.no_grad()
def encode_texts(model, tokenizer, texts, device, batch_size, max_length):
    rows = []
    model.eval()
    for start in tqdm(range(0, len(texts), batch_size), desc="encode frame definitions"):
        batch_texts = texts[start:start + batch_size]
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_special_tokens_mask=True,
            return_tensors="pt",
        )
        special_tokens_mask = encoded.pop("special_tokens_mask").to(device)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        hidden = model(**encoded, return_dict=True).last_hidden_state
        pool_mask = encoded["attention_mask"].bool() & ~special_tokens_mask.bool()
        pool_mask_float = pool_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * pool_mask_float).sum(dim=1) / pool_mask_float.sum(
            dim=1
        ).clamp_min(1.0)
        rows.append(pooled.float().cpu())
    return torch.cat(rows, dim=0)


# 解析数据路径、模型配置及运行参数；命令行传入值会覆盖这里的默认值。
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame-info", default="./dataset/frame_info.json")
    parser.add_argument("--config-file", default="/root/autodl-tmp/chinese_macbert_large/config.json")
    parser.add_argument("--vocab-file", default="/root/autodl-tmp/chinese_macbert_large/vocab.txt")
    parser.add_argument(
        "--init-checkpoint",
        default="/root/autodl-tmp/chinese_macbert_large/pytorch_model.bin",
    )
    parser.add_argument(
        "--output",
        default="./knowledge/task1_frame_semantics.pt",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


# 程序入口：Task1：将框架名称和定义编码为向量，构建框架间语义距离缓存。
def main():
    args = parse_args()
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    with open(args.frame_info, "r", encoding="utf-8") as handle:
        frame_info = json.load(handle)

    frame_names = [frame["frame_name"] for frame in frame_info]
    frame_texts = [build_frame_text(frame) for frame in frame_info]
    if len(frame_names) != len(set(frame_names)):
        raise ValueError("frame_info.json contains duplicate frame names")

    tokenizer = BertTokenizer(vocab_file=args.vocab_file, do_lower_case=True)
    config = BertConfig.from_json_file(args.config_file)
    model = BertModel(config)
    load_bert_checkpoint(model, args.init_checkpoint)
    model.to(device)

    raw_embeddings = encode_texts(
        model=model,
        tokenizer=tokenizer,
        texts=frame_texts,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    # Centering alleviates BERT embedding anisotropy before cosine similarity.
    # 先中心化再归一化，减轻预训练句向量集中在相似方向的问题。
    centered = raw_embeddings - raw_embeddings.mean(dim=0, keepdim=True)
    embeddings = F.normalize(centered, dim=-1)
    similarity = embeddings @ embeddings.T
    # 将余弦相似度转为 [0,1] 距离；数值越大表示框架语义越不相似。
    distance = ((1.0 - similarity) / 2.0).clamp(0.0, 1.0)
    distance.fill_diagonal_(0.0)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "frame_names": frame_names,
            "frame_texts": frame_texts,
            "embeddings": embeddings,
            "distance": distance,
            "pooling": "mean_without_special_tokens_then_center_and_l2_normalize",
        },
        output_path,
    )
    print(f"Saved {len(frame_names)} frame definitions to {output_path}")
    print(
        "Distance stats: "
        f"min={distance.min().item():.6f}, "
        f"mean={distance.mean().item():.6f}, "
        f"max={distance.max().item():.6f}"
    )


if __name__ == "__main__":
    main()
