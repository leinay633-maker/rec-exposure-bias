"""Ranking models (protocol §5) and the two-tower retrievers (protocol §9), in plain PyTorch."""
from __future__ import annotations

import torch
from torch import nn

from .features import CAT_FIELDS, CONTEXT_FIELDS, USER_FIELDS, VIDEO_FIELDS

USER_SIDE = ["user_id"] + USER_FIELDS
ITEM_SIDE = ["video_id", "author_id"] + VIDEO_FIELDS
RANKER_USER_SIDE = USER_SIDE + CONTEXT_FIELDS
USER_NUMERIC = ["user_id_is_click_rate", "user_id_long_view_rate", "user_id_log_count"]
ITEM_NUMERIC = ["video_id_is_click_rate", "video_id_long_view_rate", "author_id_is_click_rate",
                "author_id_long_view_rate", "video_id_log_count", "author_id_log_count"]


def mlp(sizes: list[int], dropout: float = 0.0) -> nn.Sequential:
    layers: list[nn.Module] = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        layers += [nn.Linear(a, b), nn.ReLU()]
        if dropout:
            layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class FieldEmbedding(nn.Module):
    """One table for all categorical fields, addressed through per-field offsets."""

    def __init__(self, cardinalities: list[int], dim: int):
        super().__init__()
        offsets = torch.tensor([0] + cardinalities[:-1]).cumsum(0)
        self.register_buffer("offsets", offsets, persistent=False)
        self.table = nn.Embedding(sum(cardinalities), dim)
        nn.init.normal_(self.table.weight, std=0.01)
        self.video_offset = int(offsets[CAT_FIELDS.index("video_id")])

    def forward(self, cat: torch.Tensor, fields: list[int] | None = None) -> torch.Tensor:
        if fields is None:
            return self.table(cat + self.offsets)
        return self.table(cat[:, fields] + self.offsets[fields])

    def videos(self, index: torch.Tensor) -> torch.Tensor:
        return self.table(index + self.video_offset)


def _fields(names: list[str]) -> list[int]:
    return [CAT_FIELDS.index(n) for n in names]


def history_mean(emb: FieldEmbedding, hist: torch.Tensor) -> torch.Tensor:
    mask = (hist > 0).unsqueeze(-1).float()
    vectors = emb.videos(hist) * mask
    return vectors.sum(1) / mask.sum(1).clamp(min=1.0)


class LogisticRegression(nn.Module):
    def __init__(self, cardinalities: list[int], n_num: int, **_):
        super().__init__()
        self.weights = FieldEmbedding(cardinalities, 1)
        self.linear = nn.Linear(n_num, 1)

    def forward(self, cat, num, hist):
        return self.weights(cat).sum((1, 2)) + self.linear(num).squeeze(-1)


class DCNv2(nn.Module):
    def __init__(self, cardinalities: list[int], n_num: int, dim: int = 16, layers: int = 3, **_):
        super().__init__()
        self.emb = FieldEmbedding(cardinalities, dim)
        width = len(cardinalities) * dim + n_num
        self.cross = nn.ModuleList(nn.Linear(width, width) for _ in range(layers))
        self.deep = mlp([width, 256, 128])
        self.head = nn.Linear(width + 128, 1)

    def forward(self, cat, num, hist):
        x0 = torch.cat([self.emb(cat).flatten(1), num], 1)
        x = x0
        for layer in self.cross:
            x = x0 * layer(x) + x
        return self.head(torch.cat([x, self.deep(x0)], 1)).squeeze(-1)


class DIN(nn.Module):
    def __init__(self, cardinalities: list[int], n_num: int, dim: int = 16, **_):
        super().__init__()
        self.emb = FieldEmbedding(cardinalities, dim)
        self.video_field = CAT_FIELDS.index("video_id")
        self.attention = nn.Sequential(nn.Linear(4 * dim, 64), nn.ReLU(), nn.Linear(64, 1))
        width = len(cardinalities) * dim + n_num + dim
        self.deep = mlp([width, 256, 128])
        self.head = nn.Linear(128, 1)

    def forward(self, cat, num, hist):
        fields = self.emb(cat)
        target = fields[:, self.video_field].unsqueeze(1).expand(-1, hist.shape[1], -1)
        behaviours = self.emb.videos(hist)
        weights = self.attention(torch.cat([behaviours, target, behaviours - target, behaviours * target], -1))
        weights = weights.squeeze(-1) * (hist > 0).float()  # DIN keeps raw weights, no softmax
        interest = (weights.unsqueeze(-1) * behaviours).sum(1)
        return self.head(self.deep(torch.cat([fields.flatten(1), num, interest], 1))).squeeze(-1)


class MMoE(nn.Module):
    tasks = 3

    def __init__(self, cardinalities: list[int], n_num: int, dim: int = 16, experts: int = 4, **_):
        super().__init__()
        self.emb = FieldEmbedding(cardinalities, dim)
        width = len(cardinalities) * dim + n_num
        self.experts = nn.ModuleList(mlp([width, 256, 128]) for _ in range(experts))
        self.gates = nn.ModuleList(nn.Linear(width, experts) for _ in range(self.tasks))
        self.towers = nn.ModuleList(nn.Sequential(mlp([128, 64]), nn.Linear(64, 1)) for _ in range(self.tasks))

    def forward(self, cat, num, hist):
        x = torch.cat([self.emb(cat).flatten(1), num], 1)
        expert_out = torch.stack([e(x) for e in self.experts], 1)  # (B, E, 128)
        logits = []
        for gate, tower in zip(self.gates, self.towers):
            mixed = (torch.softmax(gate(x), -1).unsqueeze(-1) * expert_out).sum(1)
            logits.append(tower(mixed).squeeze(-1))
        return torch.stack(logits, 1)  # (B, tasks)


class TwoTowerRanker(nn.Module):
    """Pointwise two-tower used as a ranker: score = <user, item> + bias, trained with BCE."""

    def __init__(self, cardinalities: list[int], n_num: int, numeric_names: list[str], dim: int = 16, **_):
        super().__init__()
        self.emb = FieldEmbedding(cardinalities, dim)
        self.user_fields, self.item_fields = _fields(RANKER_USER_SIDE), _fields(ITEM_SIDE)
        self.user_num = [numeric_names.index(n) for n in USER_NUMERIC]
        self.item_num = [numeric_names.index(n) for n in ITEM_NUMERIC]
        self.user_tower = nn.Sequential(mlp([len(self.user_fields) * dim + len(self.user_num) + dim, 256]),
                                        nn.Linear(256, 64))
        self.item_tower = nn.Sequential(mlp([len(self.item_fields) * dim + len(self.item_num), 256]),
                                        nn.Linear(256, 64))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, cat, num, hist):
        user = torch.cat([self.emb(cat, self.user_fields).flatten(1), num[:, self.user_num],
                          history_mean(self.emb, hist)], 1)
        item = torch.cat([self.emb(cat, self.item_fields).flatten(1), num[:, self.item_num]], 1)
        return (self.user_tower(user) * self.item_tower(item)).sum(-1) + self.bias


class TwoTowerRetriever(nn.Module):
    """Two-tower for retrieval: L2-normalised towers, in-batch softmax, optional logQ correction."""

    def __init__(self, cardinalities: list[int], dim: int = 16, out: int = 64, **_):
        super().__init__()
        self.emb = FieldEmbedding(cardinalities, dim)
        self.user_fields, self.item_fields = _fields(USER_SIDE), _fields(ITEM_SIDE)
        self.user_tower = nn.Sequential(mlp([len(self.user_fields) * dim + dim, 256]), nn.Linear(256, out))
        self.item_tower = nn.Sequential(mlp([len(self.item_fields) * dim, 256]), nn.Linear(256, out))

    def users(self, cat, hist):
        x = torch.cat([self.emb(cat, self.user_fields).flatten(1), history_mean(self.emb, hist)], 1)
        return nn.functional.normalize(self.user_tower(x), dim=-1)

    def items(self, cat):
        return nn.functional.normalize(self.item_tower(self.emb(cat, self.item_fields).flatten(1)), dim=-1)


RANKERS = {"logreg": LogisticRegression, "dcn": DCNv2, "din": DIN, "mmoe": MMoE, "two_tower": TwoTowerRanker}
