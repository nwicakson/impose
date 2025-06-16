################################################################################
#  igcndiff.py – DiffPose backbone with user‑specified DEQ layer indices      #
#                                                                              #
#  YAML snippet (examples)                                                     #
#  -------------------------------------------------------------------------  #
#  deq:
#    enabled: true                # master switch
#    layers: [2, 4]               # 0‑based indices to wrap with DEQ
#    default_iterations: 10
#    tolerance: 1e-4
#                                                                              #
#  • If `layers` is empty or missing, network stays explicit.                  #
#  • Indices outside 0…num_layer‑1 are ignored with a warning.                 #
################################################################################

from __future__ import absolute_import, division, print_function
import math, logging
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.ChebConv import ChebConv, _GraphConv
from models.GraFormer import MultiHeadedAttention, GraAttenLayer, GraphNet
from models.deq_wrapper import DEQAttentionBlock, DEQManager
from common.graph_utils import adj_mx_from_edges

__all__ = ["GCNdiff", "adj_mx_from_edges"]

# --------------------------------------------------------------------------- #
#  Utility functions                                                           #
# --------------------------------------------------------------------------- #

def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / (half - 1))
    emb = t.float()[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    return F.pad(emb, (0, dim % 2, 0, 0))

def swish(x):
    return x * torch.sigmoid(x)

class _ResChebGC_diff(nn.Module):
    def __init__(self, adj, in_dim, hid_dim, emd_dim):
        super().__init__()
        self.adj = adj
        self.g1 = _GraphConv(in_dim, hid_dim, 0.1)
        self.g2 = _GraphConv(hid_dim, hid_dim, 0.1)
        self.temb_proj = nn.Linear(emd_dim, hid_dim)

    def forward(self, x, temb):
        res = x
        x = self.g1(x, self.adj)
        x = x + self.temb_proj(swish(temb))[:, None, :]
        x = self.g2(x, self.adj)
        return res + x

# --------------------------------------------------------------------------- #

class GCNdiff(nn.Module):
    """DiffPose backbone where any subset of layers can be wrapped by DEQ."""

    def __init__(self, adj, cfg):
        super().__init__()
        m = cfg.model
        self.L = m.num_layer
        self.hid = m.hid_dim
        self.coords_dim = m.coords_dim
        self.n_head = m.n_head
        self.n_pts = m.n_pts
        self.emd_dim = self.hid * 4
        self.adj = adj
        logging.info(f"GCNdiff | L={self.L} | hid={self.hid} | heads={self.n_head}")

        # stem
        self.gconv_input = ChebConv(self.coords_dim[0], self.hid, K=2)

        # backbone lists
        self.atten_layers = nn.ModuleList()
        self.gconv_layers = nn.ModuleList()
        for _ in range(self.L):
            self.atten_layers.append(GraAttenLayer(
                self.hid,
                MultiHeadedAttention(self.n_head, self.hid),
                GraphNet(self.hid, self.hid, self.n_pts),
                m.dropout))
            self.gconv_layers.append(_ResChebGC_diff(adj, self.hid, self.hid, self.emd_dim))

        self.gconv_output = ChebConv(self.hid, self.coords_dim[1], K=2)

        # timestep embedding mlp
        self.temb_mlp = nn.Sequential(
            nn.Linear(self.hid, self.emd_dim), nn.SiLU(), nn.Linear(self.emd_dim, self.emd_dim)
        )

        # --- DEQ setup ---
        self.deq_mgr = DEQManager(cfg)
        self.deq_blocks: List[Optional[DEQAttentionBlock]] = [None] * self.L
        self._create_deq_blocks(cfg)
        # Legacy runner compatibility: expose .deq_block
        self.deq_block: Optional[DEQAttentionBlock] = None

    # --------------------------------------------------------------------- #
    def _create_deq_blocks(self, cfg):
        if not (hasattr(cfg, "deq") and cfg.deq.enabled):
            logging.info("DEQ disabled")
            return
        idxs = list(getattr(cfg.deq, "layers", []))
        valid = [i for i in idxs if 0 <= i < self.L]
        if len(valid) < len(idxs):
            logging.warning(f"DEQ: some indices ignored (out of range 0..{self.L-1})")
        if not valid:
            logging.warning("DEQ enabled but no valid layers provided; fallback to explicit model")
            return
        for i in valid:
            blk = DEQAttentionBlock(self.atten_layers[i], self.gconv_layers[i], f"deq_{i}", cfg)
            self.deq_blocks[i] = blk
            self.deq_mgr.register(blk)
        logging.info(f"DEQ wrapping layers {valid}")
        # -------------------------------------------------------------
        # Provide the first DEQ block via legacy attribute .deq_block
        # so older runner code (create_diffusion_model) doesn’t break.
        # If multiple DEQs requested, pick the first index in 'valid'.
        # -------------------------------------------------------------
        self.deq_block = self.deq_blocks[valid[0]]
        idxs = list(getattr(cfg.deq, "layers", []))
        valid = [i for i in idxs if 0 <= i < self.L]
        if len(valid) < len(idxs):
            logging.warning(f"DEQ: some indices ignored (out of range 0..{self.L-1})")
        if not valid:
            logging.warning("DEQ enabled but no valid layers provided; fallback to explicit model")
            return
        for i in valid:
            blk = DEQAttentionBlock(self.atten_layers[i], self.gconv_layers[i], f"deq_{i}", cfg)
            self.deq_blocks[i] = blk
            self.deq_mgr.register(blk)
        logging.info(f"DEQ wrapping layers {valid}")

    # --------------------------------------------------------------------- #
    def forward(self, x, mask, t, *_):
        temb = self.temb_mlp(timestep_embedding(t, self.hid))
        out = self.gconv_input(x, self.adj)
        for i in range(self.L):
            blk = self.deq_blocks[i]
            if blk is not None:
                out, its = blk(out, mask, temb)
                self.deq_mgr.update_stats(its)
            else:
                out = self.atten_layers[i](out, mask)
                out = self.gconv_layers[i](out, temb)
        return self.gconv_output(out, self.adj)
