# 3rd Chinese Frame Semantic Parsing Solution

> 第三届汉语框架语义解析评测解题方案  
> **A 榜排名：1 / 635 ｜ 得分：71.4551**

本仓库整理了第三届汉语框架语义解析评测的完整三阶段级联方案。整体以 **目标词感知（Target-Aware）** 为核心，在 Task 1 中引入框架语义约束，在 Task 2 中增强论元跨度召回，在 Task 3 中融合框架局部语义，并针对级联误差进行鲁棒训练。

🔗 **比赛题目：** [第三届汉语框架语义解析评测](https://tianchi.aliyun.com/competition/entrance/532338)  
📑 **完整解题报告：** [点击查看](./解题报告.pptx)

---

## 📌 任务简介

汉语框架语义解析主要包含三个子任务：

### Task 1：框架识别
给定句子和目标词，预测目标词所触发的语义框架。

### Task 2：论元识别
识别句子中与目标框架相关的论元跨度。

### Task 3：框架元素识别
对 Task 2 检测出的论元跨度进行语义角色分类。

整体流程如下：

```text
Sentence + Target
       │
       ▼
Task 1: Frame Identification
       │
       ▼
Task 2: Argument Span Detection
       │
       ▼
Task 3: Frame Element Classification
       │
       ▼
Final Frame Semantic Parse
