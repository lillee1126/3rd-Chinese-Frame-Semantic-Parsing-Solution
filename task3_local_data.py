#!/usr/bin/env python3
# Task3：角色标签体系、论元样本及流水线开发集数据处理。
# 训练读取真实框架和跨度；测试读取 Task1 预测框架和 Task2 预测跨度。
# 同一跨度若有多个真实角色，训练保留多标签；预测仍使用单个最高分角色。
import collections

import torch
from torch.utils.data import Dataset

from task2_best_data import build_marked_inputs, read_json, resolve_marker_ids


# 从框架知识库建立框架ID、全局角色ID以及框架与角色的对应关系。
class RoleSchema:
    def __init__(self, frame_info_path):
        self.frame_info = read_json(frame_info_path)
        self.frame_names = [frame["frame_name"] for frame in self.frame_info]
        self.frame_to_id = {
            name: index for index, name in enumerate(self.frame_names)
        }
        self.role_names = []
        for frame in self.frame_info:
            for role in frame["fes"]:
                if role["fe_name"] not in self.role_names:
                    self.role_names.append(role["fe_name"])
        self.role_to_id = {
            name: index for index, name in enumerate(self.role_names)
        }
        self.frame_role_mask = torch.zeros(
            len(self.frame_names), len(self.role_names), dtype=torch.bool
        )
        for frame_id, frame in enumerate(self.frame_info):
            for role in frame["fes"]:
                self.frame_role_mask[
                    frame_id, self.role_to_id[role["fe_name"]]
                ] = True


# 组织角色分类样本：训练用真实标注，测试用上游预测。
class Task3LocalDataset(Dataset):
    def __init__(
        self,
        json_path,
        frame_info_path,
        tokenizer,
        max_length=320,
        task1_path=None,
        task2_path=None,
        for_test=False,
    ):
        self.samples = read_json(json_path)
        self.schema = RoleSchema(frame_info_path)
        self.frame_names = self.schema.frame_names
        self.role_names = self.schema.role_names
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.marker_ids = resolve_marker_ids(tokenizer)
        self.for_test = for_test
        self.rows = []

        if for_test:
            if not task1_path or not task2_path:
                raise ValueError("Task3 test requires --task1-file and --task2-file")
            task1_rows = read_json(task1_path)
            task2_rows = read_json(task2_path)
            predicted_frames = {row[0]: row[1] for row in task1_rows}
            predicted_spans = collections.defaultdict(set)
            for sentence_id, start, end in task2_rows:
                predicted_spans[sentence_id].add((int(start), int(end)))
            for sample in self.samples:
                sentence_id = sample["sentence_id"]
                spans = sorted(predicted_spans.get(sentence_id, set()))
                if not spans:
                    continue
                if sentence_id not in predicted_frames:
                    raise ValueError(f"Missing Task1 frame for sentence {sentence_id}")
                frame_name = predicted_frames[sentence_id]
                if frame_name not in self.schema.frame_to_id:
                    raise ValueError(f"Unknown predicted frame: {frame_name}")
                self.rows.append(
                    {
                        "sample": sample,
                        "frame_id": self.schema.frame_to_id[frame_name],
                        "spans": [(start, end, []) for start, end in spans],
                    }
                )
        else:
            for sample in self.samples:
                frame_id = self.schema.frame_to_id[sample["frame"]]
                grouped = collections.defaultdict(set)
                for span in sample.get("cfn_spans", []):
                    start, end = int(span["start"]), int(span["end"])
                    role_id = self.schema.role_to_id[span["fe_name"]]
                    # 相同跨度的多个角色合并保存，避免数据整理时丢掉多角色标注。
                    grouped[(start, end)].add(role_id)
                    # Preserve the two known annotations whose role edge is
                    # missing from frame_info.json.
                    self.schema.frame_role_mask[frame_id, role_id] = True
                if not grouped:
                    continue
                self.rows.append(
                    {
                        "sample": sample,
                        "frame_id": frame_id,
                        "spans": [
                            (start, end, sorted(role_ids))
                            for (start, end), role_ids in sorted(grouped.items())
                        ],
                    }
                )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        sample = row["sample"]
        item = build_marked_inputs(
            sample, self.tokenizer, self.max_length, self.marker_ids
        )
        item["sentence_id"] = sample["sentence_id"]
        item["frame_id"] = row["frame_id"]
        mapped_spans = []
        for start, end, role_ids in row["spans"]:
            if not (0 <= start <= end < item["text_length"]):
                raise ValueError(
                    f"Invalid span [{start}, {end}] in sentence {sample['sentence_id']}"
                )
            mapped_spans.append(
                (
                    item["char_to_token"][start],
                    item["char_to_token"][end],
                    role_ids,
                    start,
                    end,
                )
            )
        item["spans"] = mapped_spans
        return item


# 同时补齐句子长度与跨度数量，构造角色标签和有效跨度掩码。
class Task3LocalCollator:
    def __init__(self, num_roles, pad_token_id=0, for_test=False):
        self.num_roles = num_roles
        self.pad_token_id = pad_token_id
        self.for_test = for_test

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
            "frame_ids": torch.tensor(
                [row["frame_id"] for row in rows], dtype=torch.long
            ),
            "sentence_ids": [row["sentence_id"] for row in rows],
        }
        max_spans = max(len(row["spans"]) for row in rows)
        span_starts = torch.zeros(len(rows), max_spans, dtype=torch.long)
        span_ends = torch.zeros(len(rows), max_spans, dtype=torch.long)
        span_mask = torch.zeros(len(rows), max_spans, dtype=torch.bool)
        role_targets = torch.zeros(
            len(rows), max_spans, self.num_roles, dtype=torch.float
        )
        original_spans = []
        for batch_index, row in enumerate(rows):
            original_spans.append([])
            for span_index, (start, end, role_ids, old_start, old_end) in enumerate(
                row["spans"]
            ):
                span_starts[batch_index, span_index] = start
                span_ends[batch_index, span_index] = end
                span_mask[batch_index, span_index] = True
                original_spans[-1].append((old_start, old_end))
                for role_id in role_ids:
                    role_targets[batch_index, span_index, role_id] = 1.0
        batch.update(
            {
                "span_starts": span_starts,
                "span_ends": span_ends,
                "span_mask": span_mask,
                "original_spans": original_spans,
            }
        )
        if not self.for_test:
            batch["role_targets"] = role_targets
        return batch


# 将上游预测与真实角色对齐，保留全部真实角色总数用于流水线召回率计算。
class Task3PipelineDevDataset:
    """Task1/Task2 pipeline predictions aligned with Task3 gold tuples."""

    def __init__(
        self,
        dev_path,
        frame_info_path,
        tokenizer,
        task1_path,
        task2_path,
        max_length=320,
    ):
        self.samples = read_json(dev_path)
        self.schema = RoleSchema(frame_info_path)
        self.frame_names = self.schema.frame_names
        self.role_names = self.schema.role_names
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.marker_ids = resolve_marker_ids(tokenizer)
        predicted_frames = {row[0]: row[1] for row in read_json(task1_path)}
        predicted_spans = collections.defaultdict(set)
        for sentence_id, start, end in read_json(task2_path):
            predicted_spans[sentence_id].add((int(start), int(end)))

        seen_ids = set()
        gold_tuples = set()
        self.rows = []
        for sample in self.samples:
            sentence_id = sample["sentence_id"]
            if sentence_id in seen_ids:
                continue
            seen_ids.add(sentence_id)
            gold_by_span = collections.defaultdict(set)
            for span in sample.get("cfn_spans", []):
                start, end = int(span["start"]), int(span["end"])
                role_id = self.schema.role_to_id[span["fe_name"]]
                gold_by_span[(start, end)].add(role_id)
                gold_tuples.add((sentence_id, start, end, role_id))

            spans = sorted(predicted_spans.get(sentence_id, set()))
            if not spans:
                continue
            frame_name = predicted_frames.get(sentence_id)
            if frame_name not in self.schema.frame_to_id:
                raise ValueError(
                    f"Missing or invalid predicted frame for dev sentence {sentence_id}"
                )
            self.rows.append(
                {
                    "sample": sample,
                    "frame_id": self.schema.frame_to_id[frame_name],
                    "spans": [
                        (start, end, sorted(gold_by_span.get((start, end), set())))
                        for start, end in spans
                    ],
                }
            )
        # 真实角色总数包括上游未预测出的跨度，防止流水线召回率被高估。
        self.pipeline_gold_role_count = len(gold_tuples)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        sample = row["sample"]
        item = build_marked_inputs(
            sample, self.tokenizer, self.max_length, self.marker_ids
        )
        item["sentence_id"] = sample["sentence_id"]
        item["frame_id"] = row["frame_id"]
        item["spans"] = [
            (
                item["char_to_token"][start],
                item["char_to_token"][end],
                role_ids,
                start,
                end,
            )
            for start, end, role_ids in row["spans"]
            if 0 <= start <= end < item["text_length"]
        ]
        if not item["spans"]:
            raise ValueError(
                f"All predicted spans invalid in dev sentence {sample['sentence_id']}"
            )
        return item
