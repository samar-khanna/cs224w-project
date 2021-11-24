from ogb.linkproppred import PygLinkPropPredDataset
from torch_geometric.data import DataLoader
from gnn_stack import GNNStack
from train import train

def main():
    # Download and process data at './dataset/ogbg_molhiv/'
    dataset = PygLinkPropPredDataset(name = "ogbl-ddi", root = './dataset/')
    split_idx = dataset.get_idx_split() 
    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=32, shuffle=True)
    val_loader = DataLoader(dataset[split_idx["valid"]], batch_size=32, shuffle=False)
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=32, shuffle=False)
    model = GNNStack(dataset.num_node_features, args.hidden_dim, dataset.num_classes, args)

    train(model,train_loader,val_loader)
    
if __name__=="__main__":
    main()