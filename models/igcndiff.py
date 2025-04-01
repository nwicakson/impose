from __future__ import absolute_import
from lib2to3.refactor import get_fixers_from_package

import torch.nn as nn
import torch
import numpy as np
import scipy.sparse as sp
import copy, math
import torch.nn.functional as F
import logging
from torch.nn.parameter import Parameter
from models.ChebConv import ChebConv, _GraphConv, _ResChebGC
from models.GraFormer import *
from models.deq_wrapper import DEQAttentionBlock, DEQManager

### the embedding of diffusion timestep ###
def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb

def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)
    
class _ResChebGC_diff(nn.Module):
    def __init__(self, adj, input_dim, output_dim, emd_dim, hid_dim, p_dropout):
        super(_ResChebGC_diff, self).__init__()
        self.adj = adj
        self.gconv1 = _GraphConv(input_dim, hid_dim, p_dropout)
        self.gconv2 = _GraphConv(hid_dim, output_dim, p_dropout)
        ### time embedding ###
        self.temb_proj = torch.nn.Linear(emd_dim,hid_dim)

    def forward(self, x, temb):
        residual = x
        out = self.gconv1(x, self.adj)
        out = out + self.temb_proj(nonlinearity(temb))[:, None, :]
        out = self.gconv2(out, self.adj)
        return residual + out

class GCNdiff(nn.Module):
    def __init__(self, adj, config):
        super(GCNdiff, self).__init__()
        
        self.adj = adj
        self.config = config
        ### load gcn configuration ###
        con_gcn = config.model
        self.hid_dim, self.emd_dim, self.coords_dim, num_layers, n_head, dropout, n_pts = \
            con_gcn.hid_dim, con_gcn.emd_dim, con_gcn.coords_dim, \
                con_gcn.num_layer, con_gcn.n_head, con_gcn.dropout, con_gcn.n_pts
                
        self.hid_dim = self.hid_dim
        self.emd_dim = self.hid_dim*4
        
        # Print model configuration (just once during initialization)
        logging.info(f"GCNdiff Configuration: dims={self.hid_dim}/{self.emd_dim}, layers={num_layers}, heads={n_head}")
                
        ### Generate Graphformer  ###
        self.n_layers = num_layers

        _gconv_input = ChebConv(in_c=self.coords_dim[0], out_c=self.hid_dim, K=2)
        _gconv_layers = []
        _attention_layer = []

        dim_model = self.hid_dim
        c = copy.deepcopy
        attn = MultiHeadedAttention(n_head, dim_model)
        gcn = GraphNet(in_features=dim_model, out_features=dim_model, n_pts=n_pts)

        # Create DEQ Manager to track and manage implicit iterations
        self.deq_manager = DEQManager(config)
        
        # Create standard and DEQ layers
        for i in range(num_layers):
            _attention_layer.append(GraAttenLayer(dim_model, c(attn), c(gcn), dropout))
            
            _gconv_layers.append(_ResChebGC_diff(adj=adj, input_dim=self.hid_dim, output_dim=self.hid_dim,
                emd_dim=self.emd_dim, hid_dim=self.hid_dim, p_dropout=0.1))
        
        # Apply DEQ to selected components based on config - default to disabled
        self.use_middle_layer_deq = False
        if hasattr(config, 'deq'):
            self.use_middle_layer_deq = getattr(config.deq, 'enabled', False)
            if hasattr(config.deq, 'components'):
                self.use_middle_layer_deq = self.use_middle_layer_deq and getattr(config.deq.components, 'middle_layer', False)
        
        # Log DEQ settings
        logging.info(f"GCNdiff DEQ Settings: enabled={self.use_middle_layer_deq}")
        
        mid_idx = num_layers // 2
        
        # Create DEQ block for middle layer if enabled
        if self.use_middle_layer_deq:
            self.deq_block = DEQAttentionBlock(
                _attention_layer[mid_idx], 
                _gconv_layers[mid_idx],
                name=f"mid_layer_{mid_idx}",
                config=config
            )
            # Register with the manager
            self.deq_manager.register(self.deq_block)
        
        # Store standard layers
        self.gconv_input = _gconv_input
        self.gconv_layers = nn.ModuleList(_gconv_layers)
        self.atten_layers = nn.ModuleList(_attention_layer)
        self.gconv_output = ChebConv(in_c=dim_model, out_c=self.coords_dim[1], K=2)
        
        ### diffusion configuration  ###
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            torch.nn.Linear(self.hid_dim, self.emd_dim),
            torch.nn.Linear(self.emd_dim, self.emd_dim),
        ])

    def forward(self, x, mask, t, cemd):
        # timestep embedding
        temb = get_timestep_embedding(t, self.hid_dim)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)
        
        # Initial processing
        out = self.gconv_input(x, self.adj)
        
        # First half of the network - standard processing
        for i in range(self.n_layers // 2):
            out = self.atten_layers[i](out, mask)
            out = self.gconv_layers[i](out, temb)
        
        # Middle layer with DEQ if enabled, otherwise standard processing
        mid_idx = self.n_layers // 2
        
        if self.use_middle_layer_deq and hasattr(self, 'deq_block'):
            logging.info("Using DEQ for middle layer")
            try:
                # Apply DEQ to middle layer
                mid_out, iterations = self.deq_block(out, mask, temb)
                self.deq_manager.update_stats(iterations)
                out = mid_out
                logging.info(f"DEQ iterations: {iterations}")
            except Exception as e:
                # If DEQ fails, fall back to standard processing
                logging.warning(f"DEQ failed: {e}, using standard forward pass")
                out = self.atten_layers[mid_idx](out, mask)
                out = self.gconv_layers[mid_idx](out, temb)
        else:
            # Standard processing when DEQ is disabled (original behavior)
            out = self.atten_layers[mid_idx](out, mask)
            out = self.gconv_layers[mid_idx](out, temb)
        
        # Second half of the network - standard processing
        for i in range(self.n_layers // 2 + 1, self.n_layers):
            out = self.atten_layers[i](out, mask)
            out = self.gconv_layers[i](out, temb)
        
        # Final output layer
        out = self.gconv_output(out, self.adj)
        
        return out