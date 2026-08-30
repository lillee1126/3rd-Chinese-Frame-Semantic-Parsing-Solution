#!/usr/bin/env python3
# 指标汇总工具：读取 checkpoint 中已经保存的开发集指标。
# 不会重新推理或重新评测；加权分数为 0.3×Task1准确率 + 0.3×Task2 F1 + 0.4×Task3流水线 F1。
import argparse
import json
from pathlib import Path

import torch


# 在 CPU 上读取 checkpoint，并兼容不同 PyTorch 版本的加载接口。
def load_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


# 程序入口：指标汇总工具：读取 checkpoint 中已经保存的开发集指标。
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task1", required=True)
    parser.add_argument("--task2", required=True)
    parser.add_argument("--task3", required=True)
    parser.add_argument("--task1-control")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    task1 = load_checkpoint(args.task1)
    task2 = load_checkpoint(args.task2)
    task3 = load_checkpoint(args.task3)
    task1_metrics = task1["dev_metrics"]
    task2_metrics = task2["dev_metrics"]
    task3_pipeline = task3["pipeline_dev_metrics"]
    summary = {
        "selection": {
            "task1": "dev_accuracy",
            "task2": "dev_f1",
            "task3": "pipeline_dev_f1",
        },
        "task1": {"epoch": task1["epoch"], "metrics": task1_metrics},
        "task2": {"epoch": task2["epoch"], "metrics": task2_metrics},
        "task3": {
            "epoch": task3["epoch"],
            "oracle_metrics": task3["oracle_dev_metrics"],
            "pipeline_metrics": task3_pipeline,
        },
        "weighted_pipeline_dev_score": (
            0.3 * float(task1_metrics["accuracy"])
            + 0.3 * float(task2_metrics["f1"])
            + 0.4 * float(task3_pipeline["f1"])
        ),
    }
    if args.task1_control:
        task1_control = load_checkpoint(args.task1_control)
        control_metrics = task1_control["dev_metrics"]
        summary["task1_control_weight0"] = {
            "epoch": task1_control["epoch"],
            "metrics": control_metrics,
        }
        summary["task1_semantic_accuracy_gain_over_control"] = (
            float(task1_metrics["accuracy"])
            - float(control_metrics["accuracy"])
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved dev evaluation summary: {output}")


if __name__ == "__main__":
    main()
