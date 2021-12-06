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
from utils import print_and_log


def passed_arguments():
    parser = argparse.ArgumentParser(description="Script to train online graph setting")
    parser.add_argument('--data_path', type=str,
                        default='./dataset/online_init:1000-online_nodes:10-seed:0.pkl',
                        help='Path to data .pkl file')
    parser.add_argument('--exp_dir', type=str, default=None,
                        help="Path to exp dir for model checkpoints and experiment logs")
    parser.add_argument('--init_epochs', type=int, default=100,
                        help="Number of epochs for initial subgraph training")
    parser.add_argument('--online_steps', type=int, default=10,
                        help="Number of gradient steps for online learning.")
    parser.add_argument('--init_lr', type=float, default=1e-2,
                        help="Learning rate for initial graph pre-training")
    parser.add_argument('--online_lr', type=float, default=1e-2,
                        help="Learning rate for online node learning")
    parser.add_argument('--node_dim', type=int, default=256,
                        help='Embedding dimension for nodes')
    parser.add_argument('--init_batch_size', type=int, default=1024*64,
                        help='Number of links per batch used in initial pre-training')
    parser.add_argument('--online_batch_size', type=int, default=32,
                        help='Number of links per batch used for online learning')
    parser.add_argument()
    return parser.parse_args()


if __name__ == "__main__":
    args = passed_arguments()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    hidden_dim = 32
    dropout = 0.5
    num_layers = 4
    optim_wd = 0

    init_train_epochs = args.init_epochs
    num_online_steps = args.online_steps
    init_lr = args.init_lr
    online_lr = args.online_lr
    node_emb_dim = args.node_dim
    init_train_batch_size = args.init_batch_size
    batch_size = args.online_batch_size
    path_to_dataset = args.data_path
    exp_dir = args.exp_dir
    if exp_dir is None:
        exp_dir = "./experiments"
        dir = f"online.epochs:{init_train_epochs}.online_steps:{num_online_steps}" \
              f".layers:{num_layers}.hidden_dim:{hidden_dim}.node_dim:{node_emb_dim}" \
              f".lr:{lr}.optim_wd:{optim_wd}.batch_size:{batch_size}"
        exp_dir = os.path.join(exp_dir, dir)

    model_dir = os.path.join(exp_dir, 'checkpoints')
    logs_dir = os.path.join(exp_dir, 'logs')
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    logfile_path = os.path.join(logs_dir, 'log.txt')
    if os.path.isfile(logfile_path):
        logfile = open(logfile_path, "a", buffering=1)
    else:
        logfile = open(logfile_path, "w", buffering=1)

    with open(path_to_dataset, 'rb') as f:
        dataset = pickle.load(f)

    init_nodes = dataset['init_nodes'].to(device)
    init_edge_index = dataset['init_edge_index'].to(device)
    init_pos_train = init_edge_index[:, ::2].to(device)  # Relying on interleaved order

    online_node_edge_index = dataset['online']

    emb = torch.nn.Embedding(len(init_nodes) + len(online_node_edge_index), node_emb_dim).to(device)
    model = GNNStack(node_emb_dim, hidden_dim, hidden_dim, num_layers, dropout, emb=True).to(device)
    link_predictor = LinkPredictor(hidden_dim, hidden_dim, 1, num_layers + 1, dropout).to(device)

    optimizer = optim.Adam(
        list(model.parameters()) + list(link_predictor.parameters()) + list(emb.parameters()),
        lr=init_lr, weight_decay=optim_wd
    )

    # Train on initial subgraph
    for e in range(init_train_epochs):
        loss = train(model, link_predictor, emb.weight[:len(init_nodes)], init_edge_index, init_pos_train.T,
                     init_train_batch_size, optimizer)
        print_and_log(logfile,f"Epoch {e+1}/{init_train_epochs}: Loss = {round(loss, 5)}")
        if (e + 1) % 20 == 0:
            torch.save(model.state_dict(), os.path.join(model_dir, f"init_train:{e}.pt"))

    # New optimizer for online learning
    optimizer = optim.Adam(
        list(link_predictor.parameters()) + list(emb.parameters()),
        lr=online_lr, weight_decay=optim_wd
    )

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
        curr_nodes = torch.cat((curr_nodes, torch.as_tensor([n_id], device=device)))

        # Create new embedding for n_id
        # optimizer.param_groups[0]['params'].extend(node_emb.parameters())

        # Warm start embedding for new node
        with torch.no_grad():
            emb.weight[n_id] = emb.weight[:n_id].mean(dim=0)

        # Nodes are ordered sequentially (online node ids start at len(init_nodes))
        for t in range(num_online_steps):
            loss = online_train(model, link_predictor, emb.weight[:n_id+1],
                                curr_edge_index, train_sup, train_neg, batch_size, optimizer, device)
            print_and_log(logfile,f"Step {t+1}/{num_online_steps}: loss = {round(loss, 5)}")

        torch.save(model.state_dict(), os.path.join(model_dir, f"online_id:{n_id}.pt"))

        # TODO: Is it fair to use same neg edges during train and val?
        val_tp, val_tn, val_fp, val_fn = online_eval(model, link_predictor, emb.weight[:n_id+1],
                                                     curr_edge_index, valid, valid_neg, batch_size)
        print_and_log(logfile,f"VAL accuracy: {(val_tp + val_tn)/(val_tp + val_tn + val_fp + val_fn)}")
        print_and_log(logfile,f"VAL tp: {val_tp.item()}, fn: {val_fn.item()}, tn: {val_tn.item()}, fp: {val_fp.item()}")

        test_tp, test_tn, test_fp, test_fn = online_eval(model, link_predictor, emb.weight[:n_id+1],
                                                         curr_edge_index, valid, test_neg, batch_size)

        print_and_log(logfile,f"TEST accuracy: {(test_tp + test_tn) / (test_tp + test_tn + test_fp + test_fn)}")
        print_and_log(logfile,f"TEST tp: {test_tp.item()}, fn: {test_fn.item()}, tn: {test_tn.item()}, fp: {test_fp.item()}")

    logfile.close()
