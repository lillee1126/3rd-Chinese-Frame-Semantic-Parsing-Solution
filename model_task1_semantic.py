# Task1：目标感知框架分类模型与语义损失。
# 输入原句及目标词掩码，融合整句、目标词和上下文表示，输出框架分数。
# 语义损失使用提前构建的框架距离矩阵；它只参与训练，不改变预测输出格式。
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel


# 目标感知分类器：将目标词线索注入编码器，并融合三种表示后预测框架。
class TargetAwareFrameClassifier(nn.Module):
    """Target-aware 713-way classifier with a shared, coordinate-safe input."""

    def __init__(self, config, num_labels, dropout=0.2):
        super().__init__()
        self.config = config
        self.num_labels = num_labels
        hidden_size = config.hidden_size

        self.bert = BertModel(config)
        # Index 0 is fixed at zero, so non-target tokens retain their exact
        # pretrained BERT word embeddings. Only target tokens receive a new cue.
        self.target_type_embeddings = nn.Embedding(
            2, hidden_size, padding_idx=0
        )
        with torch.no_grad():
            self.target_type_embeddings.weight[0].zero_()
            nn.init.normal_(
                self.target_type_embeddings.weight[1],
                mean=0.0,
                std=config.initializer_range,
            )

        self.target_query = nn.Linear(hidden_size, hidden_size)
        self.context_key = nn.Linear(hidden_size, hidden_size)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_size),
        )
        self.classifier = nn.Linear(hidden_size, num_labels)

    # 前向计算：目标标记嵌入 → 编码 → 目标引导上下文注意力 → 融合 → 框架分数。
    def forward(
        self,
        input_ids,
        attention_mask,
        target_mask,
        content_mask,
    ):
        # inputs_embeds replaces only BERT's word embeddings. BertEmbeddings still
        # adds its pretrained position and token-type embeddings internally.
        word_embeddings = self.bert.embeddings.word_embeddings(input_ids)
        # 仅目标位置叠加可学习提示，非目标位置保持原始词嵌入；不插入新 token。
        inputs_embeds = word_embeddings + self.target_type_embeddings(target_mask)

        output = self.bert(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
        )
        hidden = output.last_hidden_state
        # [CLS] 表示整句；下面对目标词多个字符求平均，得到目标表示。
        cls_repr = hidden[:, 0]

        target_weights = target_mask.to(hidden.dtype)
        target_length = target_weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        target_repr = (
            hidden * target_weights.unsqueeze(-1)
        ).sum(dim=1) / target_length

        # 以目标词为查询，对上下文做注意力汇聚，使模型关注与目标含义相关的内容。
        query = self.target_query(target_repr).unsqueeze(1)
        keys = self.context_key(hidden)
        attention_scores = (query * keys).sum(dim=-1) / math.sqrt(hidden.shape[-1])

        # Context pooling focuses on the sentence around the target. If a sample
        # contains only the target, fall back to allowing the target itself.
        context_mask = content_mask.bool() & ~target_mask.bool()
        empty_context = ~context_mask.any(dim=1)
        if empty_context.any():
            context_mask = context_mask.clone()
            context_mask[empty_context] = content_mask[empty_context].bool()
        attention_scores = attention_scores.masked_fill(~context_mask, -1e4)
        context_weights = torch.softmax(attention_scores, dim=-1)
        context_repr = (context_weights.unsqueeze(-1) * hidden).sum(dim=1)

        # 拼接整句、目标词、上下文三部分，再映射回隐藏维度。
        frame_feature = self.fusion(
            torch.cat([cls_repr, target_repr, context_repr], dim=-1)
        )
        logits = self.classifier(frame_feature)
        return {
            "logits": logits,
            "frame_feature": frame_feature,
            "context_weights": context_weights,
        }


# 损失函数：交叉熵加上按预测概率加权的框架语义距离代价。
class SemanticAwareFrameLoss(nn.Module):
    """CE plus expected frame-definition distance under the model distribution."""

    def __init__(
        self,
        semantic_distance,
        semantic_weight=0.2,
        distance_power=1.0,
        label_smoothing=0.05,
        class_weights=None,
    ):
        super().__init__()
        if semantic_distance.ndim != 2:
            raise ValueError("semantic_distance must be a square matrix")
        if semantic_distance.shape[0] != semantic_distance.shape[1]:
            raise ValueError("semantic_distance must be square")

        self.register_buffer("semantic_distance", semantic_distance.float())
        if class_weights is None:
            self.class_weights = None
        else:
            self.register_buffer("class_weights", class_weights.float())
        self.semantic_weight = float(semantic_weight)
        self.distance_power = float(distance_power)
        if self.distance_power <= 0:
            raise ValueError("distance_power must be greater than 0")
        self.label_smoothing = float(label_smoothing)

    # 计算 CE + 权重×语义代价期望；距离变换为 (exp(D)-1)^distance_power。
    def forward(self, logits, labels, semantic_weight=None):
        ce_per_sample = F.cross_entropy(
            logits,
            labels,
            weight=self.class_weights,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )
        probability = torch.softmax(logits.float(), dim=-1)
        # 为每个真实框架取出到全部候选框架的距离，预测越偏向远距离框架，惩罚越大。
        semantic_cost = self.semantic_distance[labels]

        # Fixed alpha=1 exponential transform followed by distance shaping:
        #   phi(D) = (exp(D) - 1) ** distance_power
        # It keeps phi(0)=0 while amplifying semantically distant errors.
        # expm1 is more numerically accurate than exp(D) - 1 near D=0.
        # expm1(D) 等价于 exp(D)-1；主方案 distance_power=2 时进一步平方。
        semantic_cost = torch.expm1(semantic_cost)
        if self.distance_power != 1.0:
            semantic_cost = semantic_cost.pow(self.distance_power)
        # 按预测概率求语义代价期望，因此该项对所有候选框架的概率均有影响。
        semantic_per_sample = (probability * semantic_cost).sum(dim=-1)

        weight = self.semantic_weight if semantic_weight is None else semantic_weight
        total_per_sample = ce_per_sample + float(weight) * semantic_per_sample
        return {
            "loss": total_per_sample.mean(),
            "ce_loss": ce_per_sample.mean(),
            "semantic_loss": semantic_per_sample.mean(),
        }
