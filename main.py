import os
import argparse
import torch
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from torch.optim import optimizer
import torch.optim as optim
from torch_geometric.data import DataLoader
from gnn_stack import GNNStack
from train import train
from link_predictor import LinkPredictor
from evaluate import test
from utils import print_and_log


def main():
    parser = argparse.ArgumentParser(description="Script to train link prediction in offline graph setting")
    parser.add_argument('--epochs', type=int, default=200,
                        help="Number of epochs for training")
    parser.add_argument('--lr', type=float, default=5e-3,
                        help="Learning rate training")
    parser.add_argument('--node_dim', type=int, default=256,
                        help='Embedding dimension for nodes')
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=64 * 1024)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--exp_dir', type=str, default=None,
                        help="Path to exp dir for model checkpoints and experiment logs")
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    optim_wd = 0
    epochs = args.epochs
    hidden_dim = args.hidden_channels
    dropout = args.dropout
    num_layers = args.num_layers
    lr = args.lr
    node_emb_dim = args.node_dim
    batch_size = args.batch_size
    exp_dir = args.exp_dir
    if exp_dir is None:
        exp_dir = "./experiments"
        dir = f"offline.epochs:{epochs}.lr{lr}.layers:{num_layers}" \
              f".hidden_dim:{hidden_dim}.node_dim:{node_emb_dim}.init_batch_size:{batch_size}"
        exp_dir = os.path.join(exp_dir, dir)

    model_dir = os.path.join(exp_dir, 'checkpoints')
    logs_dir = os.path.join(exp_dir, 'logs')
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    logfile_path = os.path.join(logs_dir, 'log.txt')
    logfile = open(logfile_path, "a" if os.path.isfile(logfile_path) else "w", buffering=1)
    
    # Download and process data at './dataset/ogbl-ddi/'
    dataset = PygLinkPropPredDataset(name="ogbl-ddi", root='./dataset/')
    split_edge = dataset.get_edge_split()
    pos_train_edge = split_edge['train']['edge'].to(device)

    graph = dataset[0]
    edge_index = graph.edge_index.to(device)

    evaluator = Evaluator(name='ogbl-ddi')

    emb = torch.nn.Embedding(graph.num_nodes, node_emb_dim).to(device)
    model = GNNStack(node_emb_dim, hidden_dim, hidden_dim, num_layers, dropout, emb=True).to(device)
    link_predictor = LinkPredictor(hidden_dim, hidden_dim, 1, num_layers + 1, dropout).to(device)

    optimizer = optim.Adam(
        list(model.parameters()) + list(link_predictor.parameters()) + list(emb.parameters()),
        lr=lr, weight_decay=optim_wd
    )

    logfile = open(logfile_path, "a" if os.path.isfile(logfile_path) else "w", buffering=1)

    for e in range(epochs):
        loss = train(model, link_predictor, emb.weight, edge_index, pos_train_edge, batch_size, optimizer)
        print_and_log(logfile,f"Epoch {e + 1}: loss: {round(loss, 5)}")

        if (e+1)%10 ==0:
            result = test(model, link_predictor, emb.weight, edge_index, split_edge, batch_size, evaluator)
            print_and_log(logfile, result)
    
    logfile.close()
if __name__ == "__main__":
    main()
