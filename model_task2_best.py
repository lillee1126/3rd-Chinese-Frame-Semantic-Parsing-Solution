#!/usr/bin/env python3
# Task2：目标增强 GlobalPointer 论元跨度模型。
# 为每一对起止位置打分，并加入可学习的软分词边界偏置和起终点辅助损失。
# 主要张量：字符表示 [批次, 序列长度, 隐藏维度]，跨度分数 [批次, 序列长度, 序列长度]。
import math

import torch
import torch.nn as nn

from transformers import BertModel


# 通过 logsumexp 稳定计算多标签损失，同时约束正例与负例分数。
def multilabel_categorical_crossentropy(labels, logits):
    """Numerically stable multilabel categorical cross-entropy."""
    transformed = (1.0 - 2.0 * labels) * logits
    negative_logits = transformed - labels * 1e12
    positive_logits = transformed - (1.0 - labels) * 1e12
    zeros = torch.zeros_like(logits[..., :1])
    negative_logits = torch.cat([negative_logits, zeros], dim=-1)
    positive_logits = torch.cat([positive_logits, zeros], dim=-1)
    return (
        torch.logsumexp(negative_logits, dim=-1)
        + torch.logsumexp(positive_logits, dim=-1)
    ).mean()


# 跨度识别器：目标条件增强、旋转位置编码、软边界和跨度联合打分。
class Task2RecallBalancedGlobalPointer(nn.Module):
    def __init__(
        self,
        config,
        inner_dim=64,
        relative_position_dim=64,
        max_relative_position=64,
        dropout=0.1,
        boundary_loss_weight=0.2,
        soft_word_boundaries=True,
    ):
        super().__init__()
        self.bert = BertModel(config)
        self.inner_dim = inner_dim
        self.max_relative_position = max_relative_position
        self.boundary_loss_weight = boundary_loss_weight
        self.soft_word_boundaries = soft_word_boundaries
        hidden_size = config.hidden_size

        self.relative_position_embeddings = nn.Embedding(
            max_relative_position * 2 + 1, relative_position_dim
        )
        self.target_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.relative_projection = nn.Linear(
            relative_position_dim, hidden_size, bias=False
        )
        self.target_gate = nn.Linear(hidden_size * 2, 1)
        self.residual_dropout = nn.Dropout(dropout)

        # The model starts as plain BERT. Target conditioning enters gradually.
        # 增强分支零初始化，避免随机目标条件在训练开始时过度扰动预训练表示。
        nn.init.zeros_(self.target_projection.weight)
        nn.init.zeros_(self.relative_projection.weight)

        self.pointer = nn.Linear(hidden_size, inner_dim * 2)
        self.start_classifier = nn.Linear(hidden_size, 1)
        self.end_classifier = nn.Linear(hidden_size, 1)

        # Word segmentation is a learnable preference, never a hard filter.
        self.word_start_bias = nn.Embedding(2, 1)
        self.word_end_bias = nn.Embedding(2, 1)
        nn.init.zeros_(self.word_start_bias.weight)
        nn.init.zeros_(self.word_end_bias.weight)

        # Boundary heads are initially auxiliary only. Their contribution to
        # span scores is learned from zero instead of disturbing GlobalPointer.
        self.start_score_scale = nn.Parameter(torch.zeros(()))
        self.end_score_scale = nn.Parameter(torch.zeros(()))

    # 计算各 token 到目标词区间的有符号距离，截断后转换为嵌入表索引。
    def relative_positions(self, target_mask):
        batch_size, sequence_length = target_mask.shape
        positions = torch.arange(sequence_length, device=target_mask.device)
        positions = positions.unsqueeze(0).expand(batch_size, -1)
        start = target_mask.float().argmax(dim=1, keepdim=True)
        reversed_mask = torch.flip(target_mask, dims=[1]).float()
        end = sequence_length - 1 - reversed_mask.argmax(dim=1, keepdim=True)
        relative = torch.where(
            positions < start,
            positions - start,
            torch.where(positions > end, positions - end, torch.zeros_like(positions)),
        )
        relative = relative.clamp(
            -self.max_relative_position, self.max_relative_position
        )
        return relative + self.max_relative_position

    # 对查询和键应用旋转位置编码，让跨度打分利用起止位置关系。
    def rope(self, tensor):
        sequence_length = tensor.shape[1]
        position = torch.arange(sequence_length, device=tensor.device).float()
        indices = torch.arange(
            0, self.inner_dim // 2, device=tensor.device
        ).float()
        indices = torch.pow(10000.0, -2.0 * indices / self.inner_dim)
        sinusoid = torch.einsum("n,d->nd", position, indices)
        sin = torch.repeat_interleave(torch.sin(sinusoid), 2, dim=-1)[None]
        cos = torch.repeat_interleave(torch.cos(sinusoid), 2, dim=-1)[None]
        rotated = torch.stack(
            [-tensor[..., 1::2], tensor[..., ::2]], dim=-1
        ).reshape_as(tensor)
        return tensor * cos + rotated * sin

    # 计算所有起止组合的分数，屏蔽非法跨度；传入标签时同时返回训练损失。
    def forward(
        self,
        input_ids,
        attention_mask,
        target_mask,
        content_mask,
        word_start_mask,
        word_end_mask,
        labels=None,
        start_labels=None,
        end_labels=None,
    ):
        hidden = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        ).last_hidden_state

        target_weights = target_mask.to(hidden.dtype)
        target_repr = (
            hidden * target_weights.unsqueeze(-1)
        ).sum(dim=1) / target_weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        target_expand = target_repr.unsqueeze(1).expand_as(hidden)
        relative_repr = self.relative_position_embeddings(
            self.relative_positions(target_mask)
        )
        target_delta = self.target_projection(target_expand)
        relative_delta = self.relative_projection(relative_repr)
        gate = torch.sigmoid(
            self.target_gate(torch.cat([hidden, target_expand], dim=-1))
        )
        # 用门控目标信息和相对位置信息增强各字符表示，保留原编码器的残差路径。
        conditioned = hidden + self.residual_dropout(
            gate * target_delta + relative_delta
        )

        start_logits = self.start_classifier(conditioned).squeeze(-1)
        end_logits = self.end_classifier(conditioned).squeeze(-1)
        # 分词边界只作为可学习偏置，不直接排除非词边界上的候选跨度。
        if self.soft_word_boundaries:
            start_logits = start_logits + self.word_start_bias(
                word_start_mask.long()
            ).squeeze(-1)
            end_logits = end_logits + self.word_end_bias(
                word_end_mask.long()
            ).squeeze(-1)

        # 将各位置投影为起点查询和终点键，计算全部起止位置组合的兼容性。
        query, key = self.pointer(conditioned).chunk(2, dim=-1)
        query = self.rope(query)
        key = self.rope(key)
        logits = torch.einsum("bmd,bnd->bmn", query, key)
        logits = logits / math.sqrt(self.inner_dim)
        logits = (
            logits
            + self.start_score_scale * start_logits.unsqueeze(2)
            + self.end_score_scale * end_logits.unsqueeze(1)
        )

        valid = content_mask.bool()
        span_mask = valid.unsqueeze(2) & valid.unsqueeze(1)
        # 只保留起点不晚于终点的组合；正文掩码同时排除特殊 token 和补齐位置。
        span_mask &= torch.ones_like(span_mask).triu()
        # 此分支仅对应硬边界消融：强制跨度起终点对齐分词边界。
        if not self.soft_word_boundaries:
            # Ablation: restore segmentation as a hard decoding/training
            # constraint. Only spans beginning/ending on annotated word
            # boundaries can receive a finite score.
            span_mask &= word_start_mask.bool().unsqueeze(2)
            span_mask &= word_end_mask.bool().unsqueeze(1)
        logits = logits.masked_fill(~span_mask, -1e4)
        masked_start_logits = start_logits.masked_fill(~valid, -1e4)
        masked_end_logits = end_logits.masked_fill(~valid, -1e4)

        loss = None
        span_loss = None
        boundary_loss = None
        if labels is not None:
            span_loss = multilabel_categorical_crossentropy(
                labels.reshape(labels.shape[0], -1),
                logits.reshape(logits.shape[0], -1),
            )
            start_loss = multilabel_categorical_crossentropy(
                start_labels, masked_start_logits
            )
            end_loss = multilabel_categorical_crossentropy(
                end_labels, masked_end_logits
            )
            # 起点与终点损失先取平均，再按 boundary_loss_weight 加入跨度损失。
            boundary_loss = 0.5 * (start_loss + end_loss)
            loss = span_loss + self.boundary_loss_weight * boundary_loss

        return {
            "logits": logits,
            "span_mask": span_mask,
            "start_logits": masked_start_logits,
            "end_logits": masked_end_logits,
            "loss": loss,
            "span_loss": span_loss,
            "boundary_loss": boundary_loss,
        }
