def online_train(model, link_predictor, emb, edge_index, pos_train_edge, neg_train_edges,
                 batch_size, optimizer, device):
    model.train()
    link_predictor.train()

    train_losses = []

    for edge_id in DataLoader(range(pos_train_edge.shape[0]), batch_size, shuffle=True):
        optimizer.zero_grad()

        node_emb = model(emb, edge_index)  # (N, d)

        pos_edge = pos_train_edge[edge_id].T  # (2, B)
        pos_pred = link_predictor(node_emb[pos_edge[0]], node_emb[pos_edge[1]])  # (B, )

        neg_idx = np.random.choice(len(neg_train_edges), edge_id.shape[0], replace=False)
        neg_edge = neg_train_edges[torch.from_numpy(neg_idx).to(device)]  # (Ne, 2)
        neg_pred = link_predictor(node_emb[neg_edge[0]], node_emb[neg_edge[1]])  # (Ne,)

        loss = -torch.log(pos_pred + 1e-15).mean() - torch.log(1 - neg_pred + 1e-15).mean()
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

    return sum(train_losses) / len(train_losses)
