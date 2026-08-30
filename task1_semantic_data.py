# Task1：数据读取、字符对齐及批次整理。
# 不插入目标标记字符，而用 target_mask 标识目标位置，保持原句坐标关系。
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


# 按 UTF-8 读取 JSON 数据。
def read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


# 兼容字典或列表形式的目标词标注，取出目标词起止位置。
def get_target_annotation(sample):
    target = sample["target"]
    if isinstance(target, list):
        if not target:
            raise ValueError(f"Empty target in sentence {sample.get('sentence_id')}")
        target = target[-1]
    return target


# 按知识库顺序建立框架标签编号，并检查框架名是否重复。
def load_frame_labels(frame_info_path):
    frame_info = read_json(frame_info_path)
    frame_names = [frame["frame_name"] for frame in frame_info]
    if len(frame_names) != len(set(frame_names)):
        raise ValueError("frame_info.json contains duplicate frame_name values")
    return frame_names, {name: index for index, name in enumerate(frame_names)}


# 按字符构造 Task1 单个样本，提供目标掩码和框架标签。
class Task1SemanticDataset(Dataset):
    """Character-aligned Task 1 data without inserting or repeating target tokens."""

    def __init__(
        self,
        json_path,
        frame_info_path,
        tokenizer,
        max_length=320,
        for_test=False,
    ):
        self.json_path = str(json_path)
        self.samples = read_json(json_path)
        self.frame_names, self.frame2id = load_frame_labels(frame_info_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.for_test = for_test

        if max_length < 4:
            raise ValueError("max_length must be at least 4")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        text = sample["text"]
        target = get_target_annotation(sample)
        target_start = int(target["start"])
        target_end = int(target["end"])

        if not (0 <= target_start <= target_end < len(text)):
            raise ValueError(
                f"Invalid target [{target_start}, {target_end}] for sentence "
                f"{sample.get('sentence_id')} with length {len(text)}"
            )

        # Chinese BERT operates character-by-character for this dataset. Building
        # the IDs explicitly preserves the original character/span coordinates.
        max_chars = self.max_length - 2
        if len(text) > max_chars:
            if target_end >= max_chars:
                raise ValueError(
                    f"Target is truncated in sentence {sample.get('sentence_id')}; "
                    f"increase --max-length above {self.max_length}"
                )
            text = text[:max_chars]

        character_ids = self.tokenizer.convert_tokens_to_ids(list(text))
        input_ids = (
            [self.tokenizer.cls_token_id]
            + character_ids
            + [self.tokenizer.sep_token_id]
        )
        attention_mask = [1] * len(input_ids)

        # Index 0 is [CLS], hence the +1 shift. No other tokens are inserted.
        # 掩码中 1 表示目标字符；由于开头有 [CLS]，原字符位置需加 1。
        target_mask = [0] * len(input_ids)
        for position in range(target_start + 1, target_end + 2):
            target_mask[position] = 1

        # Used by target-guided context attention; excludes special tokens.
        content_mask = [0] + [1] * len(text) + [0]

        item = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "target_mask": target_mask,
            "content_mask": content_mask,
            "sentence_id": sample["sentence_id"],
        }
        if not self.for_test:
            item["label"] = self.frame2id[sample["frame"]]
        return item


# 将变长 Task1 样本补齐成批次，并区分正文、目标词和填充位置。
class Task1SemanticCollator:
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
                [pad(row["attention_mask"], 0) for row in rows],
                dtype=torch.long,
            ),
            "target_mask": torch.tensor(
                [pad(row["target_mask"], 0) for row in rows],
                dtype=torch.long,
            ),
            "content_mask": torch.tensor(
                [pad(row["content_mask"], 0) for row in rows],
                dtype=torch.long,
            ),
            "sentence_ids": [row["sentence_id"] for row in rows],
        }
        if "label" in rows[0]:
            batch["labels"] = torch.tensor(
                [row["label"] for row in rows], dtype=torch.long
            )
        return batch


# 收集训练集中出现过的框架ID，供开发集分组统计准确率。
def collect_seen_frame_ids(dataset):
    return {
        dataset.frame2id[sample["frame"]]
        for sample in dataset.samples
        if "frame" in sample
    }

