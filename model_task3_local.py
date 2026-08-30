#!/usr/bin/env python3
# Task3：框架局部角色语义分类模型，同时保留全局角色分类。
# 融合论元、目标词、相对位置及框架表示，再将局部语义分数加入全局角色分数。
# 框架外角色只受到有限惩罚，不会被硬屏蔽；每个跨度在预测时选择一个最高分角色。
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel


# 角色分类器：全局角色分数与当前框架内的局部语义分数相结合。
class FrameLocalRoleClassifier(nn.Module):
    def __init__(
        self,
        config,
        frame_embeddings,
        frame_role_embeddings,
        frame_role_ids,
        num_roles,
        relative_position_dim=32,
        max_relative_position=64,
        dropout=0.2,
        initial_local_scale=5.0,
        initial_outside_penalty=4.0,
    ):
        super().__init__()
        self.bert = BertModel(config)
        self.num_frames = frame_embeddings.shape[0]
        self.num_roles = num_roles
        self.max_relative_position = max_relative_position
        hidden_size = config.hidden_size
        if frame_embeddings.shape[-1] != hidden_size:
            raise ValueError("Frame embedding dimension differs from BERT")
        if frame_role_embeddings.shape[-1] != hidden_size:
            raise ValueError("Frame-role embedding dimension differs from BERT")
        if frame_role_embeddings.shape[:2] != frame_role_ids.shape:
            raise ValueError("Frame-role semantics and IDs have different shapes")

        # Semantic tensors are supplied by the cache at construction time and
        # intentionally excluded from checkpoints to keep checkpoint size small.
        # 语义缓存注册为随设备迁移的缓冲区；persistent=False 表示不写进 checkpoint。
        self.register_buffer(
            "frame_embeddings", frame_embeddings, persistent=False
        )
        # 语义缓存注册为随设备迁移的缓冲区；persistent=False 表示不写进 checkpoint。
        self.register_buffer(
            "frame_role_embeddings", frame_role_embeddings, persistent=False
        )
        self.register_buffer("frame_role_ids", frame_role_ids, persistent=False)

        self.relative_embeddings = nn.Embedding(
            max_relative_position * 2 + 1, relative_position_dim
        )
        feature_size = hidden_size * 6 + relative_position_dim
        self.span_fusion = nn.Sequential(
            nn.Linear(feature_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_size),
        )
        self.frame_norm = nn.LayerNorm(hidden_size)
        self.frame_projection = nn.Linear(hidden_size, hidden_size)
        self.frame_gate = nn.Linear(hidden_size * 2, 1)
        nn.init.eye_(self.frame_projection.weight)
        nn.init.zeros_(self.frame_projection.bias)
        nn.init.zeros_(self.frame_gate.weight)
        nn.init.constant_(self.frame_gate.bias, -1.0)
        self.dropout = nn.Dropout(dropout)

        self.global_classifier = nn.Linear(hidden_size, num_roles)
        self.local_span_projection = nn.Linear(hidden_size, hidden_size)
        self.local_role_projection = nn.Linear(hidden_size, hidden_size)
        self.local_role_bias = nn.Embedding(num_roles, 1)
        nn.init.zeros_(self.local_role_bias.weight)
        self.log_local_scale = nn.Parameter(
            torch.tensor(math.log(initial_local_scale), dtype=torch.float)
        )
        raw_penalty = math.log(math.expm1(initial_outside_penalty))
        self.raw_outside_penalty = nn.Parameter(
            torch.tensor(raw_penalty, dtype=torch.float)
        )

    # 按批次提取指定位置的隐藏向量，用于获取论元起点和终点表示。
    @staticmethod
    def gather_positions(hidden, positions):
        index = positions.unsqueeze(-1).expand(-1, -1, hidden.shape[-1])
        return torch.gather(hidden, dim=1, index=index)

    # 用前缀和计算闭区间跨度内的平均表示，避免逐跨度循环求和。
    def span_mean_pool(self, hidden, starts, ends):
        prefix = torch.cat(
            [torch.zeros_like(hidden[:, :1]), hidden.cumsum(dim=1)], dim=1
        )
        end_sum = self.gather_positions(prefix, ends + 1)
        start_sum = self.gather_positions(prefix, starts)
        widths = (ends - starts + 1).clamp_min(1).unsqueeze(-1)
        return (end_sum - start_sum) / widths.to(hidden.dtype)

    # 计算论元与目标词区间的距离；相交时距离为零，再截断并平移为索引。
    def relative_position_ids(self, target_mask, span_starts, span_ends):
        sequence_length = target_mask.shape[1]
        target_start = target_mask.float().argmax(dim=1, keepdim=True)
        reversed_mask = torch.flip(target_mask, dims=[1]).float()
        target_end = sequence_length - 1 - reversed_mask.argmax(
            dim=1, keepdim=True
        )
        relative = torch.where(
            span_ends < target_start,
            span_ends - target_start,
            torch.where(
                span_starts > target_end,
                span_starts - target_end,
                torch.zeros_like(span_starts),
            ),
        )
        relative = relative.clamp(
            -self.max_relative_position, self.max_relative_position
        )
        return relative + self.max_relative_position

    # 为每个论元计算所有全局角色的分数，再加上当前框架内角色的语义贡献。
    def forward(
        self,
        input_ids,
        attention_mask,
        target_mask,
        frame_ids,
        span_starts,
        span_ends,
        span_mask,
        role_targets=None,
        frame_strength=None,
    ):
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        # 平均编码器最后四层表示，作为论元和目标词的基础特征。
        hidden = torch.stack(bert_output.hidden_states[-4:], dim=0).mean(dim=0)
        target_weights = target_mask.to(hidden.dtype)
        target_repr = (
            hidden * target_weights.unsqueeze(-1)
        ).sum(dim=1) / target_weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        target_expand = target_repr.unsqueeze(1).expand(
            -1, span_starts.shape[1], -1
        )
        start_repr = self.gather_positions(hidden, span_starts)
        end_repr = self.gather_positions(hidden, span_ends)
        mean_repr = self.span_mean_pool(hidden, span_starts, span_ends)
        relative_repr = self.relative_embeddings(
            self.relative_position_ids(target_mask, span_starts, span_ends)
        )
        # 融合起点、终点、跨度均值、目标均值、乘积、绝对差及相对位置特征。
        feature = torch.cat(
            [
                start_repr,
                end_repr,
                mean_repr,
                target_expand,
                mean_repr * target_expand,
                torch.abs(mean_repr - target_expand),
                relative_repr,
            ],
            dim=-1,
        )
        span_repr = self.span_fusion(feature)

        if frame_strength is None:
            frame_strength = torch.ones(
                frame_ids.shape[0], device=frame_ids.device, dtype=span_repr.dtype
            )
        else:
            frame_strength = frame_strength.to(span_repr.dtype)
        # 框架强度为零时关闭框架条件与局部先验，用于训练中的框架丢弃。
        strength_3d = frame_strength[:, None, None]

        frame_semantics = self.frame_embeddings[frame_ids].to(span_repr.dtype)
        frame_semantics = self.frame_norm(frame_semantics)
        frame_repr = self.frame_projection(frame_semantics).unsqueeze(1)
        frame_repr = frame_repr.expand_as(span_repr)
        frame_gate = torch.sigmoid(
            self.frame_gate(torch.cat([span_repr, frame_repr], dim=-1))
        )
        conditioned = span_repr + self.dropout(
            strength_3d * frame_gate * frame_repr
        )

        # 全局分类头为所有角色保留分数；局部分类只补充当前框架候选的语义证据。
        global_logits = self.global_classifier(conditioned)
        candidate_ids = self.frame_role_ids[frame_ids]
        # 局部候选用 -1 补齐；这些补齐项不能对真实角色分数产生贡献。
        candidate_valid = candidate_ids >= 0
        safe_candidate_ids = candidate_ids.clamp_min(0)
        candidate_semantics = self.frame_role_embeddings[frame_ids].to(
            conditioned.dtype
        )
        span_query = F.normalize(
            self.local_span_projection(conditioned), dim=-1
        )
        role_keys = F.normalize(
            self.local_role_projection(candidate_semantics), dim=-1
        )
        local_scale = self.log_local_scale.exp().clamp(max=20.0)
        local_logits = torch.einsum("bsh,bkh->bsk", span_query, role_keys)
        local_logits = local_logits * local_scale
        local_logits = local_logits + self.local_role_bias(
            safe_candidate_ids
        ).squeeze(-1).unsqueeze(1)

        outside_penalty = F.softplus(self.raw_outside_penalty)
        # 先施加有限的框架外惩罚，再为框架内候选抵消惩罚并补上局部语义分数。
        logits = global_logits - strength_3d * outside_penalty
        # Candidate roles cancel the outside penalty and receive their local
        # frame-specific semantic score. Padded candidates contribute zero.
        candidate_contribution = outside_penalty + local_logits
        candidate_contribution = candidate_contribution * strength_3d
        candidate_contribution = candidate_contribution * candidate_valid[
            :, None, :
        ].to(candidate_contribution.dtype)
        scatter_index = safe_candidate_ids[:, None, :].expand(
            -1, span_starts.shape[1], -1
        )
        # 按照局部到全局角色ID映射，把局部贡献累加回全局类别维度。
        logits = logits.scatter_add(2, scatter_index, candidate_contribution)

        loss = None
        if role_targets is not None:
            # 同一跨度存在多个真实角色时，将标签归一化成目标分布，再计算交叉熵。
            target_distribution = role_targets / role_targets.sum(
                dim=-1, keepdim=True
            ).clamp_min(1.0)
            per_span = -(
                target_distribution * F.log_softmax(logits, dim=-1)
            ).sum(dim=-1)
            loss = (per_span * span_mask.to(per_span.dtype)).sum()
            loss = loss / span_mask.sum().clamp_min(1)
        return {
            "logits": logits,
            "loss": loss,
            "outside_penalty": outside_penalty.detach(),
            "local_scale": local_scale.detach(),
        }
