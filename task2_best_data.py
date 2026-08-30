#!/usr/bin/env python3
# Task2：目标标记、字符坐标映射、分词边界特征及跨度标签。
# 插入 [unused1]/[unused2] 后 token 坐标发生变化，因此必须保留原字符与 token 的双向映射。
import json

import torch
from torch.utils.data import Dataset


# 按 UTF-8 读取 JSON 数据。
def read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


# 取出样本目标词列表中的最后一项，并检查是否缺少目标标注。
def get_target(sample):
    targets = sample.get("target", [])
    if not targets:
        raise ValueError(f"Missing target in sentence {sample.get('sentence_id')}")
    return targets[-1]


# 使用词表中的两个 unused token 作为目标词左右标记，并验证它们有效且不同。
def resolve_marker_ids(tokenizer):
    marker_tokens = ("[unused1]", "[unused2]")
    marker_ids = tuple(tokenizer.convert_tokens_to_ids(token) for token in marker_tokens)
    if (
        marker_ids[0] == tokenizer.unk_token_id
        or marker_ids[1] == tokenizer.unk_token_id
        or marker_ids[0] == marker_ids[1]
    ):
        raise ValueError(
            "The vocabulary must contain distinct [unused1] and [unused2] tokens"
        )
    return marker_ids


# 插入目标标记，构建掩码、分词边界特征和字符/token 坐标映射。
def build_marked_inputs(sample, tokenizer, max_length, marker_ids):
    text = sample["text"]
    target = get_target(sample)
    target_start = int(target["start"])
    target_end = int(target["end"])
    sentence_id = sample.get("sentence_id")
    if not (0 <= target_start <= target_end < len(text)):
        raise ValueError(f"Invalid target in sentence {sentence_id}")

    # [CLS], two target markers and [SEP] add four positions.
    if len(text) + 4 > max_length:
        raise ValueError(
            f"Sentence {sentence_id} needs {len(text) + 4} tokens; "
            f"increase --max-length above {max_length}"
        )

    input_ids = [tokenizer.cls_token_id]
    target_mask = [0]
    content_mask = [0]
    word_start_mask = [0]
    word_end_mask = [0]
    token_to_char = [-1]
    char_to_token = []

    start_marker_id, end_marker_id = marker_ids
    char_ids = tokenizer.convert_tokens_to_ids(list(text))
    for char_index, char_id in enumerate(char_ids):
        if char_index == target_start:
            input_ids.append(start_marker_id)
            target_mask.append(0)
            content_mask.append(0)
            word_start_mask.append(0)
            word_end_mask.append(0)
            token_to_char.append(-1)

        # 记录原字符在插入标记后的 token 位置；训练标签和预测还原都依赖此映射。
        char_to_token.append(len(input_ids))
        input_ids.append(char_id)
        target_mask.append(int(target_start <= char_index <= target_end))
        content_mask.append(1)
        word_start_mask.append(0)
        word_end_mask.append(0)
        token_to_char.append(char_index)

        if char_index == target_end:
            input_ids.append(end_marker_id)
            target_mask.append(0)
            content_mask.append(0)
            word_start_mask.append(0)
            word_end_mask.append(0)
            token_to_char.append(-1)

    input_ids.append(tokenizer.sep_token_id)
    target_mask.append(0)
    content_mask.append(0)
    word_start_mask.append(0)
    word_end_mask.append(0)
    token_to_char.append(-1)

    valid_words = 0
    for word in sample.get("word", []):
        word_start = int(word["start"])
        word_end = int(word["end"])
        if 0 <= word_start <= word_end < len(text):
            word_start_mask[char_to_token[word_start]] = 1
            word_end_mask[char_to_token[word_end]] = 1
            valid_words += 1

    # Segmentation is a soft feature, so a missing segmentation falls back to
    # treating every character as a possible word boundary.
    # 缺少合法分词标注时，将每个字符都视为可用边界，避免无边界可选。
    if valid_words == 0:
        for token_index, char_index in enumerate(token_to_char):
            if char_index >= 0:
                word_start_mask[token_index] = 1
                word_end_mask[token_index] = 1

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "target_mask": target_mask,
        "content_mask": content_mask,
        "word_start_mask": word_start_mask,
        "word_end_mask": word_end_mask,
        "char_to_token": char_to_token,
        "token_to_char": token_to_char,
        "text_length": len(text),
    }


# 读取 Task2 样本，构造带目标标记的输入及真实跨度坐标。
class Task2BestDataset(Dataset):
    def __init__(self, json_path, tokenizer, max_length=320, for_test=False):
        self.samples = read_json(json_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.for_test = for_test
        self.marker_ids = resolve_marker_ids(tokenizer)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        item = build_marked_inputs(
            sample, self.tokenizer, self.max_length, self.marker_ids
        )
        item["sentence_id"] = sample["sentence_id"]
        if not self.for_test:
            spans = set()
            for span in sample.get("cfn_spans", []):
                start = int(span["start"])
                end = int(span["end"])
                if not (0 <= start <= end < item["text_length"]):
                    raise ValueError(
                        f"Invalid gold span [{start}, {end}] in sentence "
                        f"{sample['sentence_id']}"
                    )
                spans.add(
                    (item["char_to_token"][start], item["char_to_token"][end])
                )
            item["spans"] = sorted(spans)
        return item


# 补齐输入，并构造二维跨度标签和一维起点、终点标签。
class Task2BestCollator:
    def __init__(self, pad_token_id=0):
        self.pad_token_id = pad_token_id

    def __call__(self, rows):
        max_length = max(len(row["input_ids"]) for row in rows)

        def pad(values, fill):
            return values + [fill] * (max_length - len(values))

        batch = {
            "input_ids": torch.tensor(
                [pad(row["input_ids"], self.pad_token_id) for row in rows],
                dtype=torch.long,
            ),
            "attention_mask": torch.tensor(
                [pad(row["attention_mask"], 0) for row in rows], dtype=torch.long
            ),
            "target_mask": torch.tensor(
                [pad(row["target_mask"], 0) for row in rows], dtype=torch.long
            ),
            "content_mask": torch.tensor(
                [pad(row["content_mask"], 0) for row in rows], dtype=torch.long
            ),
            "word_start_mask": torch.tensor(
                [pad(row["word_start_mask"], 0) for row in rows], dtype=torch.long
            ),
            "word_end_mask": torch.tensor(
                [pad(row["word_end_mask"], 0) for row in rows], dtype=torch.long
            ),
            "sentence_ids": [row["sentence_id"] for row in rows],
            "text_lengths": [row["text_length"] for row in rows],
            "token_to_chars": [row["token_to_char"] for row in rows],
        }

        if "spans" in rows[0]:
            # 二维标签的 [start,end] 表示一个真实论元；同时构造起终点辅助标签。
            labels = torch.zeros(
                len(rows), max_length, max_length, dtype=torch.float
            )
            start_labels = torch.zeros(len(rows), max_length, dtype=torch.float)
            end_labels = torch.zeros(len(rows), max_length, dtype=torch.float)
            for batch_index, row in enumerate(rows):
                for start, end in row["spans"]:
                    labels[batch_index, start, end] = 1.0
                    start_labels[batch_index, start] = 1.0
                    end_labels[batch_index, end] = 1.0
            batch["labels"] = labels
            batch["start_labels"] = start_labels
            batch["end_labels"] = end_labels
        return batch
