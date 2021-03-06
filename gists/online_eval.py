def online_eval(model, link_predictor, emb, edge_index, pos_edges, neg_edges, batch_size):
    """
    Evaluates model on positive and negative edges for prediction.
    1. Computes the updated node embeddings given the existing subgraph and online node's message edges
    2. Computes predictions on the positive edges, calculating:
        a. True positives: number of positive edges correctly identified as positive
        b. False negatives: number of positive edges falsely identified as negative
    3. Computes predictions on the negative edges, calculating:
        c. False positives: number of negative edges falsely identified as positive
        d. True negatives: number of negative edges correctly identified as negative
    """
    model.eval()
    link_predictor.eval()

    tp = 0.0
    tn = 0.0
    fp = 0.0
    fn = 0.0
    for edge_id in DataLoader(range(pos_edges.shape[0]), batch_size, shuffle=False, drop_last=False):
        node_emb = model(emb, edge_index)  # (N, d)

        pos_edge = pos_edges[edge_id].T  # (2, B)
        pos_pred = link_predictor(node_emb[pos_edge[0]], node_emb[pos_edge[1]]).squeeze()  # (B, )

        tp += (pos_pred >= 0.5).sum().item()
        fn += (pos_pred < 0.5).sum().item()

    for edge_id in DataLoader(range(neg_edges.shape[0]), batch_size, shuffle=False, drop_last=False):
        node_emb = model(emb, edge_index)  # (N, d)

        neg_edge = neg_edges[edge_id].T  # (2, B)
        neg_pred = link_predictor(node_emb[neg_edge[0]], node_emb[neg_edge[1]]).squeeze()  # (B, )

        fp += (neg_pred >= 0.5).sum().item()
        tn += (neg_pred < 0.5).sum().item()

    return tp, tn, fp, fn
