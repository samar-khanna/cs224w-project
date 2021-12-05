import os

import torch
import pickle
import argparse
from ogb.linkproppred import PygLinkPropPredDataset
from torch.optim import optimizer
from torch_geometric.data import DataLoader
from gnn_stack import GNNStack
from train import train
from link_predictor import LinkPredictor
import torch.optim as optim

from train import train
from online_train import online_train, online_eval


def passed_arguments():
    parser = argparse.ArgumentParser(description="Script to train online graph setting")
    parser.add_argument('--data_path', type=str,
                        default='./dataset/online_init:1000-online_nodes:10-seed:0.pkl',
                        help='Path to data .pkl file')
    parser.add_argument('--model_dir', type=str, default=None,
                        help="Path to exp dir for model weights")
    return parser.parse_args()


if __name__ == "__main__":
    args = passed_arguments()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    init_train_epochs = 2
    num_online_steps = 5
    hidden_dim = 32
    dropout = 0.5
    num_layers = 4
    lr = 1e-2
    optim_wd = 0
    node_emb_dim = 256
    batch_size = 16
    path_to_dataset = args.data_path
    model_dir = args.model_path
    if model_dir is None:
        exp_dir = "./experiments"
        model_dir = f"online.epochs:{init_train_epochs}.online_steps:{num_online_steps}" \
                     f".layers:{num_layers}.hidden_dim:{hidden_dim}.node_dim:{node_emb_dim}" \
                     f".lr:{lr}.optim_wd:{optim_wd}.batch_size:{batch_size}"
        model_dir = os.path.join(exp_dir, model_dir)
        os.makedirs(model_dir, exist_ok=True)

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
    for e in range(init_train_epochs):
        train(model, link_predictor, emb.weight[:len(init_nodes)], init_edge_index, init_pos_train,
              batch_size, optimizer)
        torch.save(model.state_dict(), os.path.join(model_dir, f"init_train:{e}.pt"))

    curr_nodes = init_nodes
    curr_edge_index = init_edge_index  # (2, E)
    for n_id, node_split in online_node_edge_index.items():
        train_msg, train_sup, train_neg, valid, valid_neg, test, test_neg = \
            node_split['train_msg'], node_split['train_sup'], node_split['train_neg'], \
            node_split['valid'], node_split['valid_neg'], node_split['test'], node_split['test_neg']

        train_msg = train_msg.to(device)
        train_sup = train_sup.to(device)
        train_neg = train_neg.to(device)
        valid = valid.to(device)
        valid_neg = valid_neg.to(device)
        test = test.to(device)
        test_neg = test_neg.to(device)

        # Add message edges to edge index
        curr_edge_index = torch.cat((curr_edge_index, train_msg.T), dim=1)  # (2, E+Tr_msg)

        # Add new node to list of curr_nodes
        curr_nodes = torch.cat((curr_nodes, torch.as_tensor([n_id])))

        # Create new embedding for n_id
        # optimizer.param_groups[0]['params'].extend(node_emb.parameters())

        # Warm start embedding for new node
        emb.weight[n_id] = emb.weight[:n_id].mean(dim=0)

        # Nodes are ordered sequentially (online node ids start at len(init_nodes))
        for t in range(num_online_steps):
            loss = online_train(model, link_predictor, emb.weight[:n_id+1],
                                curr_edge_index, train_sup, train_neg, batch_size, optimizer, device)
            print(f"Step {t+1}/{num_online_steps}: loss = {round(loss, 5)}")

        torch.save(model.state_dict(), os.path.join(model_dir, f"online_id:{n_id}.pt"))

        # TODO: Is it fair to use same neg edges during train and val?
        val_tp, val_tn, val_fp, val_fn = online_eval(model, link_predictor, emb.weight[:n_id+1],
                                                     curr_edge_index, valid, valid_neg, batch_size)
        print(f"VAL accuracy: {(val_tp + val_tn)/(val_tp + val_tn + val_fp + val_fn)}")
        print(f"VAL tp: {val_tp.item()}, fn: {val_fn.item()}, tn: {val_tn.item()}, fp: {val_fp.item()}")

        test_tp, test_tn, test_fp, test_fn = online_eval(model, link_predictor, emb.weight[:n_id+1],
                                                         curr_edge_index, valid, test_neg, batch_size)

        print(f"TEST accuracy: {(test_tp + test_tn) / (test_tp + test_tn + test_fp + test_fn)}")
        print(f"TEST- tp: {test_tp.item()}, fn: {test_fn.item()}, tn: {test_tn.item()}, fp: {test_fp.item()}")
