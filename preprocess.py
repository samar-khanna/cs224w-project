import torch
import pickle
import numpy as np
import torch_geometric as pyg
from ogb.linkproppred import PygLinkPropPredDataset


class NoEdgeException(Exception):
    pass


def create_online_edge_index(n_id, full_edge_index, curr_edge_index, curr_nodes,
                             train_msg=0.4, train_sup=0.4, val_pct=0.1):
    curr_edges = curr_edge_index.T  # (CE, 2)
    edges = full_edge_index.T  # (E, 2)

    # First search for all edges containing node id
    # since undirected, both (i,j) and (j, i) should be in edges
    all_node_edges = edges[edges[:, 0] == n_id]  # (D_all, 2)

    # Then, only keep edges from node_id to nodes in current graph
    node_edges = torch.isin(all_node_edges[:, 1], curr_nodes)  # (D_all,)
    node_edges = all_node_edges[node_edges]  # (D, 2)
    D = node_edges.shape[0]

    # Then, split node edges into train/val/test
    train_msg_range = (0, int(train_msg*D))
    train_sup_range = (train_msg_range[1], train_msg_range[1] + int(train_sup*D))
    val_range = (train_sup_range[1], train_sup_range[1] + int(val_pct*D))
    test_range = (val_range[1], D)

    split = {
        'train_msg': node_edges[train_msg_range[0]:train_msg_range[1]],  # (TrMsg, 2)
        'train_sup': node_edges[train_sup_range[0]:train_sup_range[1]],  # (TrSup, 2)
        'valid': node_edges[val_range[0]:val_range[1]],   # (Val, 2)
        'test': node_edges[test_range[0]:test_range[1]]  # (Test, 2)
    }

    # Msg edges need both (i,j) and (j,i)
    split['train_msg'] = torch.cat((split['train_msg'], split['train_msg'].flip(1)), dim=0)

    for k, edges in split.items():
        if len(edges) == 0:
            raise NoEdgeException(f"Warning: node {n_id} has no {k} edges")
            # print(f"Warning: node {n_id} has no {k} edges")

    return torch.cat((curr_edges, split['train_msg']), dim=0).T, \
           torch.cat((curr_nodes, torch.as_tensor([n_id]))), \
           split


def preprocess(outfile, init_cluster_size=1000, num_online=None, seed=0):
    rng = np.random.default_rng(seed)

    dataset = PygLinkPropPredDataset(name="ogbl-ddi", root='./dataset/')
    split_edge = dataset.get_edge_split()

    graph = dataset[0]
    edge_index = graph.edge_index.T  # (TrE, 2)

    # All train edges are in edge_index. None of val or test edges are in edge_index
    val_edges = torch.cat((split_edge['valid']['edge'], split_edge['valid']['edge'].flip(1)), dim=0)
    test_edges = torch.cat((split_edge['test']['edge'], split_edge['test']['edge'].flip(1)), dim=0)
    full_index = torch.cat((edge_index, val_edges, test_edges), dim=0)  # (E, 2)

    nodes = np.arange(graph.num_nodes)
    node_map = np.arange(len(nodes))
    rng.shuffle(node_map)

    # old_to_new[i] = new idx of node i in new ordering of nodes
    new_from_old = torch.from_numpy(node_map[node_map])
    old_from_new = torch.from_numpy(node_map)

    # Map edges to new ordering of nodes (where new node 0 = node_map[0])
    full_index = new_from_old[full_index].T  # (2, E)

    # Initial node induced subgraph of all
    init_nodes = torch.arange(init_cluster_size)
    init_edge_index, _ = pyg.utils.subgraph(init_nodes, full_index)  # (2, InitEdges)

    num_online = num_online if num_online is not None else len(nodes) - init_cluster_size
    online_nodes = torch.arange(init_cluster_size, init_cluster_size + num_online)

    # For online nodes, find edges that connect node to current subgraph.
    # Add the online node's training message edges to the current subgraph to update the curr edge_index
    # Add the node's training, val, test edges to the online node dictionary
    curr_nodes = init_nodes
    curr_edge_index = init_edge_index
    online_node_edge_index = {}
    for n in online_nodes.numpy():
        try:
            curr_edge_index, curr_nodes, node_split = \
                create_online_edge_index(n, full_index, curr_edge_index, curr_nodes)
        except NoEdgeException as e:
            print(str(e))
            continue
        online_node_edge_index[n] = node_split

    # Save the graph info
    dataset = {
        "init_nodes": init_nodes,
        "init_edge_index": init_edge_index,
        "online": online_node_edge_index,
        "full_edge_index": full_index,
    }

    with open(outfile, 'wb') as f:
        pickle.dump(dataset, f)
