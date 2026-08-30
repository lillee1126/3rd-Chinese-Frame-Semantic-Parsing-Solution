#!/usr/bin/env python3
# Task3：编码框架及框架内角色定义，生成局部语义缓存。
# 缓存同时保存框架向量、局部角色向量、局部到全局角色ID映射；预测时也需要该缓存。
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import BertConfig, BertModel, BertTokenizer

from build_task1_frame_semantics import encode_texts, load_bert_checkpoint


# 解析数据路径、模型配置及运行参数；命令行传入值会覆盖这里的默认值。
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame-info", default="./dataset/frame_info.json")
    parser.add_argument("--config-file", default="/root/autodl-tmp/chinese_macbert_large/config.json")
    parser.add_argument("--vocab-file", default="/root/autodl-tmp/chinese_macbert_large/vocab.txt")
    parser.add_argument(
        "--init-checkpoint", default="/root/autodl-tmp/chinese_macbert_large/pytorch_model.bin"
    )
    parser.add_argument("--output", default="./knowledge/task3_local_semantics_macbert_large.pt")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


# 构造单个框架的名称与定义文本。
def frame_text(frame):
    return (
        f"框架名称：{frame['frame_name']}。"
        f"框架定义：{str(frame.get('frame_def', '')).strip()}"
    )


# 组合框架上下文和角色定义，保留同名角色的多条定义。
def frame_role_text(frame, roles):
    name = roles[0]["fe_name"]
    definitions = []
    for role in roles:
        definition = str(role.get("fe_def", "")).strip()
        if definition and definition not in definitions:
            definitions.append(definition)
    return (
        f"框架名称：{frame['frame_name']}。"
        f"框架定义：{str(frame.get('frame_def', '')).strip()}。"
        f"框架元素名称：{name}。"
        f"框架元素定义：{'；'.join(definitions)}"
    )


# 程序入口：Task3：编码框架及框架内角色定义，生成局部语义缓存。
def main():
    args = parse_args()
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    with open(args.frame_info, "r", encoding="utf-8") as handle:
        frame_info = json.load(handle)
    frame_names = [frame["frame_name"] for frame in frame_info]
    role_names = []
    for frame in frame_info:
        for role in frame["fes"]:
            if role["fe_name"] not in role_names:
                role_names.append(role["fe_name"])
    role_to_id = {name: index for index, name in enumerate(role_names)}
    maximum_roles = max(
        len({role["fe_name"] for role in frame["fes"]})
        for frame in frame_info
    )

    local_texts = []
    occurrence_locations = []
    frame_role_ids = torch.full(
        (len(frame_info), maximum_roles), -1, dtype=torch.long
    )
    frame_role_mask = torch.zeros(
        len(frame_info), len(role_names), dtype=torch.bool
    )
    for frame_id, frame in enumerate(frame_info):
        # A small number of frame_info rows repeat the same FE name, sometimes
        # with a second definition. They represent one output label, so merge
        # them into one frame-local candidate while retaining all definitions.
        # 合并同一框架内同名角色，保留多条定义，但输出标签只保留一个。
        grouped_roles = {}
        for role in frame["fes"]:
            grouped_roles.setdefault(role["fe_name"], []).append(role)
        for local_id, (name, occurrences) in enumerate(grouped_roles.items()):
            global_id = role_to_id[name]
            # 记录局部候选到全局角色编号的映射，供模型汇总局部分数。
            frame_role_ids[frame_id, local_id] = global_id
            frame_role_mask[frame_id, global_id] = True
            local_texts.append(frame_role_text(frame, occurrences))
            occurrence_locations.append((frame_id, local_id))

    tokenizer = BertTokenizer(vocab_file=args.vocab_file, do_lower_case=True)
    config = BertConfig.from_json_file(args.config_file)
    model = BertModel(config)
    load_bert_checkpoint(model, args.init_checkpoint)
    model.to(device)
    frames = encode_texts(
        model,
        tokenizer,
        [frame_text(frame) for frame in frame_info],
        device,
        args.batch_size,
        args.max_length,
    )
    local = encode_texts(
        model,
        tokenizer,
        local_texts,
        device,
        args.batch_size,
        args.max_length,
    )
    frames = F.normalize(frames - frames.mean(dim=0, keepdim=True), dim=-1)
    local = F.normalize(local - local.mean(dim=0, keepdim=True), dim=-1)

    hidden_size = local.shape[-1]
    frame_role_embeddings = torch.zeros(
        len(frame_info), maximum_roles, hidden_size, dtype=torch.float
    )
    for embedding, (frame_id, local_id) in zip(local, occurrence_locations):
        frame_role_embeddings[frame_id, local_id] = embedding

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "frame_names": frame_names,
            "role_names": role_names,
            "frame_embeddings": frames.half(),
            "frame_role_embeddings": frame_role_embeddings.half(),
            "frame_role_ids": frame_role_ids,
            "frame_role_mask": frame_role_mask,
            "maximum_roles": maximum_roles,
            "role_occurrences": len(local_texts),
            "pooling": "frame-specific FE text; mean pool; center; L2 normalize",
        },
        output_path,
    )
    print(
        f"Saved {len(frame_names)} frames, {len(role_names)} global labels and "
        f"{len(local_texts)} frame-specific FE definitions to {output_path}; "
        f"maximum_roles={maximum_roles}"
    )


if __name__ == "__main__":
    main()
