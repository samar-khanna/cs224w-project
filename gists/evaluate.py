def test(model, predictor, emb, edge_index, split_edge, batch_size, evaluator):
    """
    Evaluates model on positive and negative test edges
    1. Computes the updated node embeddings given the edge index (i.e. the message passing edges)
    2. Computes predictions on the positive and negative edges
    3. Calculates hits @ k given predictions using the ogb evaluator
    """
    model.eval()
    predictor.eval()

    node_emb = model(emb, edge_index)

    pos_test_edge = split_edge['test']['edge'].to(emb.device)
    neg_test_edge = split_edge['test']['edge_neg'].to(emb.device)

    pos_test_preds = []
    for perm in DataLoader(range(pos_test_edge.size(0)), batch_size):
        edge = pos_test_edge[perm].t()
        pos_test_preds += [predictor(node_emb[edge[0]], node_emb[edge[1]]).squeeze().cpu()]
    pos_test_pred = torch.cat(pos_test_preds, dim=0)

    neg_test_preds = []
    for perm in DataLoader(range(neg_test_edge.size(0)), batch_size):
        edge = neg_test_edge[perm].t()
        neg_test_preds += [predictor(node_emb[edge[0]], node_emb[edge[1]]).squeeze().cpu()]
    neg_test_pred = torch.cat(neg_test_preds, dim=0)

    results = {}
    for K in [20, 50, 100]:
        evaluator.K = K #using the Evaluator function in the ogb.linkproppred package
        test_hits = evaluator.eval({
            'y_pred_pos': pos_test_pred,
            'y_pred_neg': neg_test_pred,
        })[f'hits@{K}']

        results[f'Hits@{K}'] = test_hits

    return results