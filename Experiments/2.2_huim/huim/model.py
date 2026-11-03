"""Compact HUIM-inspired Stage 1 click model (an adaptation, not a reproduction).

Shared article tower -> candidate-conditioned (DIN-style) attention over the
user's fixed-history items -> matching features [h, c, h-c, h*c] -> fused with a
separate request/pair feature branch (concatenation or learned gate) -> MLP ->
click logit. No recurrence and no positional encoding: permuting aligned
history items (with their own timestamps/engagement) cannot change the output.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import torch
from torch import nn
import torch.nn.functional as F

from .features import PAIR_DENSE_COLUMNS, pair_dense

ATTENTION_MODES = ("softmax", "unnormalised", "uniform")
FUSION_MODES = ("concat", "gated")


@dataclass
class ModelConfig:
    n_category: int
    n_article_type: int
    n_topic: int
    n_subcategory: int
    n_sentiment: int
    n_read_bin: int
    n_scroll_bin: int
    n_recency_bin: int
    n_request_dense: int
    n_pair_dense: int
    text_dim: int
    image_dim: int
    d_model: int = 64
    text_proj: int = 32
    image_proj: int = 32
    attention_mode: str = "softmax"
    attention_hidden: tuple = (32, 16)
    fusion: str = "concat"
    feature_hidden: int = 64
    head_hidden: tuple = (128, 64)
    dropout: float = 0.2
    modality_dropout: float = 0.15
    recency_edges_h: list = field(default_factory=list)

    def to_json(self) -> dict:
        d = asdict(self)
        d["attention_hidden"] = list(self.attention_hidden)
        d["head_hidden"] = list(self.head_hidden)
        return d

    @classmethod
    def from_json(cls, d: dict) -> "ModelConfig":
        d = dict(d)
        d["attention_hidden"] = tuple(d["attention_hidden"])
        d["head_hidden"] = tuple(d["head_hidden"])
        return cls(**d)


def mlp(sizes, dropout, final_activation=False):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2 or final_activation:
            layers += [nn.ReLU(), nn.Dropout(dropout)]
    return nn.Sequential(*layers)


class ArticleEncoder(nn.Module):
    """Shared article representation for candidates and history items.

    Frozen inputs: L2-normalised text (gemini-embedding-2) and image (EB-NeRD)
    vectors, projected by *separate* trained linear maps (different spaces; no
    text-image similarity is computed). Trained: metadata embeddings and fusion.
    Missing vectors contribute zero plus an explicit has_* flag."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.category = nn.Embedding(cfg.n_category, 16)
        self.article_type = nn.Embedding(cfg.n_article_type, 8)
        self.sentiment = nn.Embedding(cfg.n_sentiment, 4)
        self.topics = nn.EmbeddingBag(cfg.n_topic, 16, mode="mean", padding_idx=0)
        self.subcats = nn.EmbeddingBag(cfg.n_subcategory, 8, mode="mean", padding_idx=0)
        self.text = nn.Linear(cfg.text_dim, cfg.text_proj) if cfg.text_dim else None
        self.image = nn.Linear(cfg.image_dim, cfg.image_proj) if cfg.image_dim else None
        in_dim = 16 + 8 + 4 + 16 + 8 + 8 + (cfg.text_proj if cfg.text_dim else 0) + (cfg.image_proj if cfg.image_dim else 0)
        self.out = nn.Sequential(nn.Linear(in_dim, cfg.d_model), nn.LayerNorm(cfg.d_model))

    def forward(self, art_t: dict, idx: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        parts = [self.category(art_t["category"][idx]), self.article_type(art_t["article_type"][idx]),
                 self.sentiment(art_t["sentiment"][idx])]
        tp, sp = art_t["topics_pad"][idx], art_t["subcats_pad"][idx]
        # EmbeddingBag(mean) over an all-padding row returns zeros (no topics).
        parts += [self.topics(tp), self.subcats(sp)]
        has_text = art_t["has_text"][idx]
        has_image = art_t["has_image"][idx]
        if self.training and cfg.modality_dropout > 0:
            # Modality dropout: the test catalogue is mostly without vectors, so
            # the model must also learn to score from metadata alone.
            has_text = has_text & (torch.rand(has_text.shape, device=idx.device) >= cfg.modality_dropout)
            has_image = has_image & (torch.rand(has_image.shape, device=idx.device) >= cfg.modality_dropout)
        dense = torch.stack([art_t["premium"][idx], art_t["premium_missing"][idx], art_t["body_z"][idx],
                             art_t["body_missing"][idx], art_t["topic_count_z"][idx],
                             has_text.float(), has_image.float(), art_t["known"][idx].float()], dim=1)
        parts.append(dense)
        if self.text is not None:
            ptr = art_t["text_ptr"][idx].clamp_min(0)
            vec = art_t["text_vectors"][ptr].float() * has_text[:, None]
            parts.append(self.text(vec) * has_text[:, None])
        if self.image is not None:
            ptr = art_t["image_ptr"][idx].clamp_min(0)
            vec = art_t["image_vectors"][ptr].float() * has_image[:, None]
            parts.append(self.image(vec) * has_image[:, None])
        # Unknown article (index 0): metadata indices are 0/missing, vectors absent and
        # known=0, so it maps to one learned "unknown article" representation.
        return self.out(torch.cat(parts, dim=1))


class CandidateAttention(nn.Module):
    """DIN-style target attention a_j = MLP([x_j, c, x_j - c, x_j * c]).

    The first layer is written as A x_j + B c + P (x_j * c); this is the same
    function class as a Linear layer on the 4-way concatenation (since
    W[x, c, x-c] = (Wx + Wd) x + (Wc - Wd) c) but lets catalogue scoring reuse the
    per-request A x_j and per-article B c terms."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d, hidden = cfg.d_model, list(cfg.attention_hidden)
        self.mode = cfg.attention_mode
        self.A = nn.Linear(d, hidden[0])
        self.B = nn.Linear(d, hidden[0], bias=False)
        self.P = nn.Linear(d, hidden[0], bias=False)
        self.rest = mlp(hidden + [1], dropout=0.0)
        self.empty = nn.Parameter(torch.zeros(d))
        self.norm = nn.LayerNorm(d) if self.mode == "unnormalised" else nn.Identity()

    def forward(self, x: torch.Tensor, ax: torch.Tensor, mask: torch.Tensor, c: torch.Tensor, bc: torch.Tensor):
        """x/ax: [P,H,d]/[P,H,a] history (gathered per pair); mask [P,H]; c/bc: [P,d]/[P,a].
        Returns pooled history h [P,d] and weights [P,H]."""
        any_valid = mask.any(dim=1, keepdim=True)
        if self.mode == "uniform":  # control: no candidate conditioning
            w = mask.float() / mask.float().sum(dim=1, keepdim=True).clamp_min(1.0)
            h = (w.unsqueeze(-1) * x).sum(1)
            return torch.where(any_valid, h, self.empty.expand_as(h)), w
        hidden = F.relu(ax + bc.unsqueeze(1) + self.P(x * c.unsqueeze(1)))
        score = self.rest(hidden).squeeze(-1)  # [P,H]
        if self.mode == "softmax":
            score = score.float().masked_fill(~mask, float("-inf"))
            score = score - score.max(dim=1, keepdim=True).values.masked_fill(~any_valid, 0.0)
            w = torch.exp(score) * mask
            w = w / w.sum(dim=1, keepdim=True).clamp_min(1e-12)
            w = torch.where(any_valid, w, torch.zeros_like(w))
            h = (w.unsqueeze(-1).to(x.dtype) * x).sum(1)
        else:  # "unnormalised": HUIM/DIN raw masked sum (+ LayerNorm for scale; our addition)
            w = score * mask
            h = self.norm((w.unsqueeze(-1) * x).sum(1))
        return torch.where(any_valid, h, self.empty.expand_as(h).to(h.dtype)), w


class HUIMStage1(nn.Module):
    def __init__(self, cfg: ModelConfig, request_mean, request_std, request_scaled, pair_mean, pair_std, pair_scaled):
        super().__init__()
        if cfg.attention_mode not in ATTENTION_MODES or cfg.fusion not in FUSION_MODES:
            raise ValueError("unknown attention/fusion mode")
        self.cfg = cfg
        d = cfg.d_model
        self.articles = ArticleEncoder(cfg)
        self.read_bin = nn.Embedding(cfg.n_read_bin, 8)
        self.scroll_bin = nn.Embedding(cfg.n_scroll_bin, 8)
        self.recency_bin = nn.Embedding(cfg.n_recency_bin, 8)
        self.event = nn.Sequential(nn.Linear(d + 24, d), nn.LayerNorm(d))
        self.attention = CandidateAttention(cfg)
        self.match_proj = nn.Sequential(nn.Linear(4 * d, d), nn.ReLU())
        n_dense = cfg.n_request_dense + cfg.n_pair_dense
        self.features = mlp([n_dense, cfg.feature_hidden, cfg.feature_hidden], cfg.dropout, final_activation=True)
        fused = d + cfg.feature_hidden
        self.fuse_norm = nn.LayerNorm(fused)
        self.gate = mlp([fused, fused, fused], 0.0) if cfg.fusion == "gated" else None
        self.head = mlp([fused, *cfg.head_hidden, 1], cfg.dropout)
        for name, value in (("request_mean", request_mean), ("request_std", request_std),
                            ("pair_mean", pair_mean), ("pair_std", pair_std)):
            self.register_buffer(name, torch.as_tensor(value, dtype=torch.float32))
        self.register_buffer("request_scaled", torch.as_tensor(request_scaled, dtype=torch.bool))
        self.register_buffer("pair_scaled", torch.as_tensor(pair_scaled, dtype=torch.bool))
        self.register_buffer("recency_edges_s",
                             torch.as_tensor([e * 3600.0 for e in cfg.recency_edges_h], dtype=torch.float32))

    # ------------------------------------------------------------- pieces --
    def request_state(self, ctx_t: dict, art_t: dict, req_pos: torch.Tensor) -> dict:
        """Per-request history encoding, reused across every scored article."""
        user = ctx_t["req_user"][req_pos]
        idx = ctx_t["hist_idx"][user]                     # [R,H]
        mask = ctx_t["hist_mask"][user]
        R, H = idx.shape
        uniq, inv = torch.unique(idx.reshape(-1), return_inverse=True)
        e = self.articles(art_t, uniq)[inv].reshape(R, H, -1)
        age_s = (ctx_t["req_s"][req_pos][:, None] - ctx_t["hist_s"][user]).float()
        rec = torch.bucketize(age_s, self.recency_edges_s, right=True) + 1
        rec = torch.where(mask, rec, torch.zeros_like(rec))
        ev = torch.cat([e, self.read_bin(ctx_t["hist_read_bin"][user]), self.scroll_bin(ctx_t["hist_scroll_bin"][user]),
                        self.recency_bin(rec)], dim=-1)
        x = self.event(ev)
        raw = ctx_t["req_dense_raw"][req_pos]
        dense = torch.where(self.request_scaled, (raw - self.request_mean) / self.request_std, raw)
        return {"x": x, "ax": self.attention.A(x), "mask": mask, "dense": dense}

    def article_state(self, art_t: dict, art_idx: torch.Tensor) -> dict:
        c = self.articles(art_t, art_idx)
        return {"c": c, "bc": self.attention.B(c)}

    def score_from_states(self, ctx_t, art_t, n_articles, req_pos, art_idx, rs, rpos, ast, apos, return_weights=False):
        """Logits for aligned pairs; rs/ast are request/article states and
        rpos/apos index into them for each pair."""
        x, ax, mask = rs["x"][rpos], rs["ax"][rpos], rs["mask"][rpos]
        c, bc = ast["c"][apos], ast["bc"][apos]
        h, w = self.attention(x, ax, mask, c, bc)
        match = self.match_proj(torch.cat([h, c, h - c, h * c], dim=1))
        raw_pair = pair_dense(ctx_t, art_t, req_pos, art_idx, n_articles)
        pair = torch.where(self.pair_scaled, (raw_pair - self.pair_mean) / self.pair_std, raw_pair)
        f = self.features(torch.cat([rs["dense"][rpos], pair], dim=1))
        z = self.fuse_norm(torch.cat([match, f], dim=1))
        if self.gate is not None:
            z = torch.sigmoid(self.gate(z)) * z * 2.0  # learned gate around 1 at init
        logit = self.head(z).squeeze(-1)
        return (logit, w) if return_weights else logit

    def forward(self, ctx_t: dict, art_t: dict, n_articles: int, req_pos: torch.Tensor, art_idx: torch.Tensor,
                return_weights: bool = False):
        """Aligned pairs (training / recorded slates)."""
        ureq, rpos = torch.unique(req_pos, return_inverse=True)
        uart, apos = torch.unique(art_idx, return_inverse=True)
        rs = self.request_state(ctx_t, art_t, ureq)
        ast = self.article_state(art_t, uart)
        return self.score_from_states(ctx_t, art_t, n_articles, req_pos, art_idx, rs, rpos, ast, apos, return_weights)


def build_model(cfg: ModelConfig, pre) -> HUIMStage1:
    rm, rs = pre.dense_scaler("request")
    pm, ps = pre.dense_scaler("pair")
    req_cols = pre.state["request_scaler"]["columns"]
    rmask = [c in set(pre.state["request_scaled_columns"]) for c in req_cols]
    pmask = [c in set(pre.state["pair_scaled_columns"]) for c in PAIR_DENSE_COLUMNS]
    return HUIMStage1(cfg, rm, rs, rmask, pm, ps, pmask)


def parameter_counts(model: nn.Module) -> dict:
    groups = {}
    for name, p in model.named_parameters():
        top = name.split(".")[0]
        groups[top] = groups.get(top, 0) + p.numel()
    groups["total_trainable"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return groups
