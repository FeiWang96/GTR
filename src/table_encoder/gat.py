# -*- coding: utf-8 -*-
"""
@Author     : Fei Wang
@Contact    : fwang1412@gmail.com
@Time       : 2020/12/9 22:20
@Description: 
"""
from torch import nn
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv


class GATEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, layer_num=4, hidden_dim=300, num_heads=4, feat_drop=0.1, attn_drop=0.1,
                 residual=True, activation=F.leaky_relu):
        super().__init__()

        assert output_dim % 4 == 0
        assert layer_num > 1

        layers = []
        norms = []

        first_layer = GATConv(in_feats=input_dim, out_feats=hidden_dim, num_heads=num_heads, feat_drop=feat_drop,
                              attn_drop=attn_drop, residual=residual, activation=activation)

        first_norm = nn.Sequential(
            nn.Linear(hidden_dim * num_heads, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        layers.append(first_layer)
        norms.append(first_norm)

        for _ in range(layer_num - 2):
            middle_layer = GATConv(in_feats=hidden_dim, out_feats=hidden_dim, num_heads=num_heads,
                                   feat_drop=feat_drop, attn_drop=attn_drop, residual=residual, activation=activation)

            middle_norm = nn.Sequential(
                nn.Linear(hidden_dim * num_heads, hidden_dim),
                nn.LayerNorm(hidden_dim),
            )
            layers.append(middle_layer)
            norms.append(middle_norm)

        last_layer = GATConv(in_feats=hidden_dim, out_feats=output_dim, num_heads=num_heads,
                             feat_drop=feat_drop, attn_drop=attn_drop, residual=residual, activation=activation)
        last_norm = nn.Sequential(
            nn.Linear(output_dim * num_heads, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        layers.append(last_layer)
        norms.append(last_norm)

        self.layers = nn.ModuleList(layers)
        self.norms = nn.ModuleList(norms)

    def forward(self, g, features):
        x = features
        for layer, norm in zip(self.layers, self.norms):
            x = layer(g, x).view(features.shape[0], -1)
            x = norm(x)
        return x
