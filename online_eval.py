import torch
import pickle
import numpy as np
import copy
from tqdm import trange
from torch_geometric.data import DataLoader
from utils import print_and_log

def online_eval(model, link_predictor, emb, edge_index, pos_edges, neg_edges, batch_size,res_file):
    model.eval()
    link_predictor.eval()

    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for edge_id in DataLoader(range(pos_edges.shape[0]), batch_size, shuffle=False, drop_last=False):
        node_emb = model(emb, edge_index)  # (N, d)

        pos_edge = pos_edges[edge_id].T  # (2, B)
        pos_pred = link_predictor(node_emb[pos_edge[0]], node_emb[pos_edge[1]])  # (B, )

        tp += (pos_pred >= 0.5).sum()
        fn += (pos_pred < 0.5).sum()

    for edge_id in DataLoader(range(neg_edges.shape[0]), batch_size, shuffle=False, drop_last=False):
        node_emb = model(emb, edge_index)  # (N, d)

        neg_edge = neg_edges[edge_id].T  # (2, B)
        neg_pred = link_predictor(node_emb[neg_edge[0]], node_emb[neg_edge[1]])  # (B, )

        fp += (neg_pred >= 0.5).sum()
        tn += (neg_pred < 0.5).sum()

    if res_file is not None:
        print_and_log(res_file,f"corr_pred = {pos_edge.T[pos_pred >= 0.5]} \ninc_pred = {neg_edge.T[neg_pred >= 0.5]}")
    return tp, tn, fp, fn