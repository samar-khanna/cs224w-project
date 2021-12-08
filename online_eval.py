import torch
from torch_geometric.data import DataLoader


def online_eval(model, link_predictor, emb, edge_index, pos_edges, neg_edges, batch_size):
    model.eval()
    link_predictor.eval()

    tp = 0.
    tn = 0.
    fp = 0.
    fn = 0.

    tp_pred = torch.empty(0, dtype=pos_edges.dtype)
    fp_pred = torch.empty(0, dtype=pos_edges.dtype)
    fn_pred = torch.empty(0, dtype=pos_edges.dtype)
    for edge_id in DataLoader(range(pos_edges.shape[0]), batch_size, shuffle=False, drop_last=False):
        node_emb = model(emb, edge_index)  # (N, d)

        pos_edge = pos_edges[edge_id].T  # (2, B)
        pos_pred = link_predictor(node_emb[pos_edge[0]], node_emb[pos_edge[1]]).squeeze()  # (B, )

        tp += (pos_pred >= 0.5).sum().item()
        fn += (pos_pred < 0.5).sum().item()

        tp_pred = torch.cat((tp_pred, pos_edge.T[pos_pred >= 0.5].cpu()), dim=0)
        fn_pred = torch.cat((fn_pred, pos_edge.T[pos_pred < 0.5].cpu()), dim=0)

    for edge_id in DataLoader(range(neg_edges.shape[0]), batch_size, shuffle=False, drop_last=False):
        node_emb = model(emb, edge_index)  # (N, d)

        neg_edge = neg_edges[edge_id].T  # (2, B)
        neg_pred = link_predictor(node_emb[neg_edge[0]], node_emb[neg_edge[1]]).squeeze()  # (B, )

        fp += (neg_pred >= 0.5).sum().item()
        tn += (neg_pred < 0.5).sum().item()

        # Don't care about tn coz those are too many
        fp_pred = torch.cat((fp_pred, neg_edge.T[neg_pred >= 0.5].cpu()), dim=0)

    preds = {'tp_pred': tp_pred, 'fp_pred': fp_pred, 'fn_pred': fn_pred}
    return tp, tn, fp, fn, preds
