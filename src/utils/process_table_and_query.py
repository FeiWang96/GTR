# -*- coding: utf-8 -*-
"""
@Author     : Fei Wang
@Contact    : fwang1412@gmail.com
@Time       : 2020/12/28 19:46
@Description: 
"""
from tqdm import tqdm
import numpy as np
import torch


def process_tables(tables, constructor):
    for tid in tqdm(tables.keys(), desc="processing tables"):
        dgl_graph, node_features, table_data = constructor.construct_graph(tables[tid])
        tables[tid]["dgl_graph"] = dgl_graph
        if isinstance(node_features, np.ndarray):
            node_features = torch.FloatTensor(node_features)
        tables[tid]["node_features"] = node_features


def process_queries(queries, constructor):
    features = {}
    for qid in tqdm(queries.keys(), desc="processing queries"):
        feature = constructor.w2v[" ".join(queries[qid])]
        if isinstance(feature, np.ndarray):
            feature = torch.FloatTensor(feature)
        features[qid] = feature
    return features
