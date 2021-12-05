import torch
import pickle
from ogb.linkproppred import PygLinkPropPredDataset
from torch.optim import optimizer
from torch_geometric.data import DataLoader
from gnn_stack import GNNStack
from train import train
from link_predictor import LinkPredictor
import torch.optim as optim

from train import train
from online_train import online_train, online_eval


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    init_train_epochs = 100
    num_online_steps = 5
    hidden_dim = 32
    dropout = 0.5
    num_layers = 4
    lr = 1e-2
    optim_wd = 0
    node_emb_dim = 256
    batch_size = 16
    path_to_dataset = './dataset/online_init:1000_online:10.pkl'

    with open(path_to_dataset, 'rb') as f:
        dataset = pickle.load(f)

    init_nodes = dataset['init_nodes']
    init_edge_index = dataset['init_edge_index']
    init_pos_train = init_edge_index[:, ::2]  # Relying on order

    online_node_edge_index = dataset['online']

    emb = torch.nn.Embedding(len(init_nodes) + len(online_node_edge_index), node_emb_dim).to(device)
    model = GNNStack(node_emb_dim, hidden_dim, hidden_dim, num_layers, dropout, emb=True).to(device)
    link_predictor = LinkPredictor(hidden_dim, hidden_dim, 1, num_layers + 1, dropout).to(device)

    optimizer = optim.Adam(
        list(model.parameters()) + list(link_predictor.parameters()) + list(emb.parameters()),
        lr=lr, weight_decay=optim_wd
    )

    # Train on initial subgraph
    # for e in range(init_train_epochs):
    #     train(model, link_predictor, emb.weight[:len(init_nodes)], init_edge_index, init_pos_train,
    #           batch_size, optimizer)

    curr_nodes = init_nodes
    curr_edge_index = init_edge_index  # (2, E)
    for n_id, node_split in online_node_edge_index.items():
        train_msg, train_sup, valid, test = \
            node_split['train_msg'], node_split['train_sup'], node_split['valid'], node_split['test']

        print(train_msg.shape)
        print(train_sup.shape)
        print(valid.shape)
        print(test.shape)

        # Add message edges to edge index
        curr_edge_index = torch.cat((curr_edge_index, train_msg.T), dim=1)  # (2, E+Tr_msg)

        # Create negative edges
        anc_neg_edges = []
        for n in curr_nodes:
            if not torch.isin(n, torch.cat((train_msg, train_sup, valid, test), dim=0)):
                anc_neg_edges.append((n_id, n))
        anc_neg_edges = torch.as_tensor(anc_neg_edges, dtype=torch.long).to(device)
        print(anc_neg_edges.shape)

        # Add new node to list of curr_nodes
        curr_nodes = torch.cat((curr_nodes, torch.as_tensor([n_id])))

        # Create new embedding for n_id
        # optimizer.param_groups[0]['params'].extend(node_emb.parameters())

        # Nodes are ordered sequentially (online node ids start at len(init_nodes))
        for t in range(num_online_steps):
            loss = online_train(model, link_predictor, emb.weight[:n_id+1],
                                curr_edge_index, train_sup, anc_neg_edges, batch_size, optimizer, device)
            print(f"Step {t+1}/{num_online_steps}: loss = {round(loss, 5)}")

        # TODO: Is it fair to use same neg edges during train and val?
        val_tp, val_tn, val_fp, val_fn = online_eval(model, link_predictor, emb.weight[:n_id+1],
                                                     curr_edge_index, valid, anc_neg_edges[:100], batch_size)
        print(f"Val accuracy: {(val_tp + val_tn)/(val_tp + val_tn + val_fp + val_fn)}")
        print(val_tp, val_tn, val_fp, val_fn)

        # test_tp, test_tn, test_fp, test_fn = online_eval(model, link_predictor, emb.weight[:n_id+1],
        #                                                  curr_edge_index, valid, anc_neg_edges, batch_size)


        # print(f"Test accuracy: {(test_tp + test_tn) / (test_tp + test_tn + test_fp + test_fn)}")
