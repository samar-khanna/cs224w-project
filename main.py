import torch
from ogb.linkproppred import PygLinkPropPredDataset
from torch.optim import optimizer
from torch_geometric.data import DataLoader
from gnn_stack import GNNStack
from train import train
from link_predictor import LinkPredictor
import torch.optim as optim


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    hidden_dim = 32
    dropout = 0.5
    num_layers = 4
    lr = 1e-2
    optim_wd = 0
    node_emb_dim = 256
    batch_size = 64*1024

    # Download and process data at './dataset/ogbl-ddi/'
    dataset = PygLinkPropPredDataset(name="ogbl-ddi", root='./dataset/')
    split_edge = dataset.get_edge_split()

    graph = dataset[0]
    edge_index = graph.edge_index.to(device)

    # train_loader = DataLoader(dataset[split_edge["train"]], batch_size=32, shuffle=True)
    # val_loader = DataLoader(dataset[split_edge["valid"]], batch_size=32, shuffle=False)
    # test_loader = DataLoader(dataset[split_edge["test"]], batch_size=32, shuffle=False)

    emb = torch.nn.Embedding(graph.num_nodes, node_emb_dim).to(device)
    model = GNNStack(node_emb_dim, hidden_dim, hidden_dim, num_layers, dropout, emb=True).to(device)
    link_predictor = LinkPredictor(hidden_dim, hidden_dim, 1, num_layers + 1, dropout).to(device)

    optimizer = optim.Adam(
        list(model.parameters()) + list(link_predictor.parameters()) + list(emb.parameters()),
        lr=lr, weight_decay=optim_wd
    )
    train(model, link_predictor, emb.weight, edge_index, split_edge, batch_size, optimizer, device)


if __name__ == "__main__":
    main()
