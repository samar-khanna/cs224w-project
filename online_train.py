import torch
import pickle
import numpy as np
import copy
from tqdm import trange
from torch_geometric.data import DataLoader
from torch_geometric.utils import negative_sampling


def online_train(model, link_predictor, emb, edge_index, pos_train_edge, neg_train_edges,
                 batch_size, optimizer, device):
    model.train()
    link_predictor.train()

    train_losses = []
    val_accs = []
    best_acc = 0
    best_model = None
    scheduler = None

    for edge_id in DataLoader(range(pos_train_edge.shape[0]), batch_size, shuffle=True):
        optimizer.zero_grad()

        node_emb = model(emb, edge_index)  # (N, d)

        pos_edge = pos_train_edge[edge_id].T  # (2, B)
        pos_pred = link_predictor(node_emb[pos_edge[0]], node_emb[pos_edge[1]])  # (B, )

        neg_idx = np.random.choice(len(neg_train_edges), batch_size, replace=False)
        neg_edge = neg_train_edges[torch.from_numpy(neg_idx).to(device)]  # (Ne, 2)
        neg_pred = link_predictor(node_emb[neg_edge[0]], node_emb[neg_edge[1]])  # (Ne,)

        loss = -torch.log(pos_pred + 1e-15).mean() - torch.log(1 - neg_pred + 1e-15).mean()
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())
        print(loss.item())

    return sum(train_losses) / len(train_losses)


def online_eval(model, link_predictor, emb, edge_index, pos_edges, neg_edges, batch_size):
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

    return tp, tn, fp, fn
