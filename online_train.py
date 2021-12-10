import torch
import numpy as np
from torch_geometric.data import DataLoader


def online_train(model, link_predictor, emb, edge_index, pos_train_edge, neg_train_edges,
                 batch_size, optimizer, device):
    """
    Runs training for a single online node given its edges to the existing subgraph
    :param model: Torch Graph model used for updating node embeddings based on message passing
    :param link_predictor: Torch model used for predicting whether edge exists or not
    :param emb: (N+1, d) Initial node embeddings for all N nodes in subgraph, along with new online node
    :param edge_index: (2, E) Edge index for edges in subgraph, along with message edges for online node
    :param pos_train_edge: (PE, 2) Positive edges from online node to subgraph, for supervision loss
    :param neg_train_edges: (NE, 2) All training negative edges from online node to subgraph.
        (Equal number of negative edges will be sampled as the number of positive edges for batch)
    :param batch_size: Number of positive (and negative) supervision edges to sample per batch
    :param optimizer: Torch Optimizer to update model parameters
    :param device: PyTorch device
    :return: Average supervision loss over all positive (and correspondingly sampled negative) edges
    """
    model.train()
    link_predictor.train()

    train_losses = []

    for edge_id in DataLoader(range(pos_train_edge.shape[0]), batch_size, shuffle=True):
        optimizer.zero_grad()

        # Run message passing on the inital node embeddings to get updated embeddings
        node_emb = model(emb, edge_index)  # (N, d)

        # Predict the class probabilities on the batch of positive edges using link_predictor
        pos_edge = pos_train_edge[edge_id].T  # (2, B)
        pos_pred = link_predictor(node_emb[pos_edge[0]], node_emb[pos_edge[1]])  # (B, )

        # Here we are given negative edges, so sample same number as pos edges and predict probabilities
        neg_idx = np.random.choice(len(neg_train_edges), edge_id.shape[0], replace=False)
        neg_edge = neg_train_edges[torch.from_numpy(neg_idx).to(device)]  # (Ne, 2)
        neg_pred = link_predictor(node_emb[neg_edge[0]], node_emb[neg_edge[1]])  # (Ne,)

        # Compute the corresponding negative log likelihood loss on the positive and negative edges
        loss = -torch.log(pos_pred + 1e-15).mean() - torch.log(1 - neg_pred + 1e-15).mean()
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

    return sum(train_losses) / len(train_losses)
