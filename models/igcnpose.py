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

class GCNpose(nn.Module):
    def __init__(self, adj, config):
        super(GCNpose, self).__init__()
        
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
        logging.info(f"GCNpose Configuration: dims={self.hid_dim}/{self.emd_dim}, layers={num_layers}, heads={n_head}")
                
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
        
        # Create standard layers
        for i in range(num_layers):
            _attention_layer.append(GraAttenLayer(dim_model, c(attn), c(gcn), dropout))
            _gconv_layers.append(_ResChebGC(adj=adj, input_dim=self.hid_dim, output_dim=self.hid_dim,
                                            hid_dim=self.hid_dim, p_dropout=0.1))
        
        # Apply DEQ to selected components based on config - default to disabled
        self.use_deq_refinement = False
        if hasattr(config, 'deq'):
            self.use_deq_refinement = getattr(config.deq, 'enabled', False)
            if hasattr(config.deq, 'components'):
                self.use_deq_refinement = self.use_deq_refinement and getattr(config.deq.components, 'final_layer', False)
        
        # Log DEQ settings
        logging.info(f"GCNpose DEQ Settings: use_deq_refinement={self.use_deq_refinement}")
        
        # Flag for best epoch
        self.is_best_epoch = False
        
        # Create DEQ block for final layer refinement if enabled
        if self.use_deq_refinement:
            self.deq_refinement = DEQAttentionBlock(
                _attention_layer[-1],
                _gconv_layers[-1],
                name="final_refinement",
                config=config
            )
            self.deq_manager.register(self.deq_refinement)
        
        # Store standard layers
        self.gconv_input = _gconv_input
        self.gconv_layers = nn.ModuleList(_gconv_layers)
        self.atten_layers = nn.ModuleList(_attention_layer)
        self.gconv_output = ChebConv(in_c=dim_model, out_c=3, K=2)

    def forward(self, x, mask):
        # Initial processing
        out = self.gconv_input(x, self.adj)
        
        # Process through all but the last layer normally
        for i in range(self.n_layers - 1):
            out = self.atten_layers[i](out, mask)
            out = self.gconv_layers[i](out)
        
        # For the last layer, we can use DEQ refinement when needed
        if self.use_deq_refinement and hasattr(self, 'is_best_epoch') and self.is_best_epoch:
            logging.info("Using DEQ for final layer refinement")
            try:
                # Apply DEQ to final layer for refinement
                refined_out, iterations = self.deq_refinement(out, mask)
                self.deq_manager.update_stats(iterations)
                out = refined_out
                logging.info(f"DEQ iterations: {iterations}")
            except Exception as e:
                # Fall back to standard processing
                logging.warning(f"DEQ refinement failed: {e}, using standard forward pass")
                out = self.atten_layers[-1](out, mask)
                out = self.gconv_layers[-1](out)
        else:
            # Standard processing for last layer (original behavior)
            out = self.atten_layers[-1](out, mask)
            out = self.gconv_layers[-1](out)
            
        # Final output layer
        out = self.gconv_output(out, self.adj)
        
        return out