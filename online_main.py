import os
import torch
import pickle
import argparse
import torch.optim as optim
from gnn_stack import GNNStack
from link_predictor import LinkPredictor
from torch_geometric.data import DataLoader
from ogb.linkproppred import PygLinkPropPredDataset

from train import train
from online_train import online_train
from online_eval import online_eval
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
    parser.add_argument('--init_batch_size', type=int, default=1024 * 64,
                        help='Number of links per batch used in initial pre-training')
    parser.add_argument('--online_batch_size', type=int, default=32,
                        help='Number of links per batch used for online learning')
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
    init_batch_size = args.init_batch_size
    online_batch_size = args.online_batch_size
    path_to_dataset = args.data_path
    exp_dir = args.exp_dir

    # Get dataset
    with open(path_to_dataset, 'rb') as f:
        dataset = pickle.load(f)

    init_nodes = dataset['init_nodes'].to(device)
    init_edge_index = dataset['init_edge_index'].to(device)
    init_pos_train = init_edge_index[:, ::2].to(device)  # Relying on interleaved order

    online_node_edge_index = dataset['online']

    # Configure experiment saving directories
    if exp_dir is None:
        exp_dir = "./experiments"
        dir = f"online.init_nodes:{len(init_nodes)}.num_online:{len(online_node_edge_index)}.{path_to_dataset.split('-')[2]}" \
              f".epochs:{init_train_epochs}.online_steps:{num_online_steps}" \
              f".layers:{num_layers}.hidden_dim:{hidden_dim}.node_dim:{node_emb_dim}" \
              f".init_lr:{init_lr}.online_lr:{online_lr}.optim_wd:{optim_wd}" \
              f".init_batch_size:{init_batch_size}.online_batch_size:{online_batch_size}"
        exp_dir = os.path.join(exp_dir, dir)

    model_dir = os.path.join(exp_dir, 'checkpoints')
    logs_dir = os.path.join(exp_dir, 'logs')
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    logfile_path = os.path.join(logs_dir, 'log.txt')
    resfile_val_path = os.path.join(logs_dir, 'res_val.pkl')
    resfile_test_path = os.path.join(logs_dir, 'res_test.pkl')
    logfile = open(logfile_path, "a" if os.path.isfile(logfile_path) else "w", buffering=1)

    # Create embedding, model, and optimizer
    emb = torch.nn.Embedding(len(init_nodes) + max(online_node_edge_index) + 1, node_emb_dim).to(device)
    model = GNNStack(node_emb_dim, hidden_dim, hidden_dim, num_layers, dropout, emb=True).to(device)
    link_predictor = LinkPredictor(hidden_dim, hidden_dim, 1, num_layers + 1, dropout).to(device)

    optimizer = optim.Adam(
        list(model.parameters()) + list(link_predictor.parameters()) + list(emb.parameters()),
        lr=init_lr, weight_decay=optim_wd
    )

    # Train on initial subgraph
    for e in range(init_train_epochs):
        loss = train(model, link_predictor, emb.weight[:len(init_nodes)], init_edge_index, init_pos_train.T,
                     init_batch_size, optimizer)
        print_and_log(logfile, f"Epoch {e + 1}/{init_train_epochs}: Loss = {round(loss, 5)}")
        if (e + 1) % 20 == 0:
            torch.save(model.state_dict(), os.path.join(model_dir, f"init_train:{e}.pt"))

    # New optimizer for online learning (don't update GraphSAGE)
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
            emb.weight[n_id] = emb.weight[curr_nodes].mean(dim=0)

        # Nodes are ordered sequentially (online node ids start at len(init_nodes))
        for t in range(num_online_steps):
            loss = online_train(model, link_predictor, emb.weight[:n_id + 1],
                                curr_edge_index, train_sup, train_neg, online_batch_size, optimizer, device)
            print_and_log(logfile, f"Step {t + 1}/{num_online_steps}: loss = {round(loss, 5)}")

        torch.save(model.state_dict(), os.path.join(model_dir, f"online_id:{n_id}_model.pt"))
        torch.save(emb.state_dict(), os.path.join(model_dir, f"online_id:{n_id}_emb.pt"))
        torch.save(link_predictor.state_dict(), os.path.join(model_dir, f"online_id:{n_id}_lp.pt"))
        val_res = {}
        test_res = {}
        val_tp, val_tn, val_fp, val_fn, preds = online_eval(model, link_predictor, emb.weight[:n_id + 1],
                                                     curr_edge_index, valid, valid_neg, online_batch_size)
        val_res[n_id] = preds

        print_and_log(logfile,f"For node {n_id}")
        print_and_log(logfile, f"VAL accuracy: {(val_tp + val_tn) / (val_tp + val_tn + val_fp + val_fn)}")
        print_and_log(logfile, f"VAL tp: {val_tp}, fn: {val_fn}, tn: {val_tn}, fp: {val_fp}")

        test_tp, test_tn, test_fp, test_fn, preds = online_eval(model, link_predictor, emb.weight[:n_id + 1],
                                                         curr_edge_index, valid, test_neg, online_batch_size,)
        test_res[n_id] = preds
        print_and_log(logfile, f"TEST accuracy: {(test_tp + test_tn) / (test_tp + test_tn + test_fp + test_fn)}")
        print_and_log(logfile, f"TEST tp: {test_tp}, fn: {test_fn}, tn: {test_tn}, fp: {test_fp}")
    
    with open(resfile_val_path, 'wb') as f:
        pickle.dump(val_res, f)
    with open(resfile_test_path, 'wb') as f:
        pickle.dump(test_res, f)
    logfile.close()
    
