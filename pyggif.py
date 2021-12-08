from genericpath import exists
import pickle

import torch

print(torch.__version__)
import torch_geometric

print(torch_geometric.__version__)
import networkx as nx
import matplotlib.pyplot as plt
from pylab import show
import collections
import imageio
import os

def tuplify(tensor):
    return (int(tensor[0].item()),int(tensor[1].item()))
    
def listify(tensor):
    l = []
    for t in tensor:
        l.append(tuplify(t))
    return l

def visualize(base_graph_edges, base_extra_nodes, new_nodes, new_given_edges,pred_new_node_edges, correct_new_edge, figsize,gif_name,framerate=24):
    G_skel = nx.Graph()
    G_skel.add_edges_from(base_graph_edges)
    for edge_list in pred_new_node_edges:
        G_skel.add_edges_from(edge_list)
    for edge_list in correct_new_edge:
        G_skel.add_edges_from(edge_list)
    for edge_list in new_given_edges:
        G_skel.add_edges_from(edge_list) 
    pos = nx.spring_layout(G_skel)

    min_x_pos = 10000
    max_x_pos = -10000
    min_y_pos = 10000
    max_y_pos = -10000

    for k,v in pos.items():
        min_x_pos = min(min_x_pos,v[0])
        min_y_pos = min(min_y_pos,v[1])
        max_x_pos = max(max_x_pos,v[0])
        max_y_pos = max(max_y_pos,v[1])

    print(min_x_pos,max_x_pos,min_y_pos,max_y_pos)

    min_x_pos -=0.5
    min_y_pos -=0.5
    max_x_pos +=0.5
    max_y_pos +=0.5

    filenames = []
    # BASE GRAPH
    G_old = nx.Graph()
    G_old.add_edges_from(base_graph_edges)
    G_old.add_nodes_from(base_extra_nodes)
    plt.figure(figsize=figsize)
    plt.xlim(min_x_pos,max_x_pos)
    plt.ylim(min_y_pos,max_y_pos)
    pos_old = {i:pos[i] for i in G_old.nodes()}
    node_labels_old = {i:i for i in G_old.nodes()}
    node_color_old = ['b' for node in G_old.nodes()]

    nx.draw(G_old, pos=pos_old,node_color=node_color_old, labels=node_labels_old, font_color='white')
    filename = f'{0}.png'
    filenames.append(filename)
    plt.savefig(filename)

    G_pred = nx.Graph()
    G_pred.clear()
    G_pred.add_edges_from(base_graph_edges)
    G_pred.add_nodes_from(base_extra_nodes)
    edge_color = {}
    edge_weight = {}
    # for edge in G_pred.edges():
    #   edge_color[edge] = 'black'
    pos_new = {i:pos[i] for i in G_pred.nodes()}
    node_labels = {i:i for i in G_pred.nodes()}
    # Iterating over new nodes
    edges_learnt = []
    for index, new_node in enumerate(new_nodes):
        G_pred.add_edges_from(pred_new_node_edges[index])
        G_pred.add_edges_from(correct_new_edge[index])
        G_pred.add_edges_from(new_given_edges[index])
        pos_new[new_node] = pos[new_node]
        node_labels[new_node] = new_node
        for edge in G_pred.edges():
            rev_edge = edge[::-1]
            if edge[1] != new_node:
                if edge in base_graph_edges or rev_edge in edges_learnt:
                    edge_color[edge] = 'black'
                    edge_weight[edge] = 1
                elif rev_edge in pred_new_node_edges[index] and rev_edge in correct_new_edge[index]:
                    edge_color[edge] = 'purple'
                    edge_weight[edge] = 2
                    edges_learnt.append(rev_edge)
                else:
                    edge_color[edge] = 'white'
                    edge_weight[edge] = 1
                    # print('whiten',edge)
            if rev_edge in new_given_edges[index]:
                edge_color[edge] = 'blue'
                edge_weight[edge] = 2
                edges_learnt.append(rev_edge)
                # print("given ",edge)
            elif rev_edge in correct_new_edge[index] and rev_edge in pred_new_node_edges[index] :
                edge_color[edge] = 'green'
                edge_weight[edge] = 5
                edges_learnt.append(rev_edge)
                print(edge, "is correct")
            elif rev_edge in pred_new_node_edges[index]:
                edge_color[edge] = 'red'
                edge_weight[edge] = 5
                print(edge, "is wrong")
            elif rev_edge in correct_new_edge[index]:
                edge_weight[edge] = 3
                edge_color[edge] = 'yellow'
                # print(edge, "was missed")
      # print('edges learnt so far')
      # print(edges_learnt)
      # print('-------')
        node_color = ['r' if node==new_node else 'b' for node in G_pred.nodes()]
        edges = G_pred.edges()
        ec = [edge_color[edge] for edge in edges]
        ew = [edge_weight[edge] for edge in edges]
        plt.figure(figsize=figsize)
        plt.xlim(min_x_pos,max_x_pos)
        plt.ylim(min_y_pos,max_y_pos)
        nx.draw(G_pred, pos=pos_new, labels=node_labels, edge_color=ec, width=ew, node_color=node_color, font_color='white')
        filename = f'{index+1}.png'
        filenames.append(filename)
        plt.savefig(filename)

    with imageio.get_writer(gif_name, mode='I') as writer:
        for filename in filenames:
            for _ in range(framerate):
                image = imageio.imread(filename)
                writer.append_data(image)

    for filename in set(filenames):
        os.remove(filename)


if __name__ == "__main__":
    # ! python preprocess.py --init_size 75 --num_online 10
    # ! python online_main.py --data_path /content/cs224w-project/dataset/online_init:75-online_nodes:10-split:0.4_0.4_0.1_0.1-seed:0.pkl
    res_val_path = './experiments/online.init_nodes:75.num_online:7.online_nodes:10.epochs:100.online_steps:10.layers:4.hidden_dim:32.node_dim:256.init_lr:0.01.online_lr:0.01.optim_wd:0.init_batch_size:65536.online_batch_size:32/logs/res_val.pkl'
    res_val = pickle.load(open(res_val_path, 'rb'))

    with open('./dataset/online_init:75-online_nodes:10-split:0.4_0.4_0.1_0.1-seed:0.pkl', 'rb') as f:
        dataset = pickle.load(f)
    online_node_edge_index = dataset['online']
    new_nodes = list(res_val.keys())[:3]
    print(f"New Nodes: {new_nodes}")
    new_given_edges = []
    correct_new_edge = []
    all_online_edges = torch.Tensor()
    for n_id in new_nodes:
        t_msg = online_node_edge_index[n_id]['train_msg'][1::2]
        t_sup = online_node_edge_index[n_id]['train_sup']
        new_given_edges.append(listify(torch.cat([t_msg,t_sup],dim=0)))

        valid = online_node_edge_index[n_id]['valid']
        test = online_node_edge_index[n_id]['test']
        correct_new_edge.append(listify(torch.cat([valid,test],dim=0)))
        all_online_edges = torch.cat([all_online_edges,t_msg,t_sup,valid,test],dim=0)
    print("num all_onl_edges = ",all_online_edges.shape)

    pred_new_node_edges = []
    for n_id in new_nodes:
        pred_new_node_edges.append(listify(torch.cat([res_val[n_id]['corr_pred'].view(-1,2),res_val[n_id]['inc_pred'].view(-1,2)],dim=0)))
    print("num_pred_e=",len(pred_new_node_edges))

    for p in pred_new_node_edges:
        all_online_edges = torch.cat([all_online_edges, torch.Tensor(p)],dim=0)
    all_online_edges = all_online_edges.type(dtype=torch.int64)
    print("num all_onl_edges = ",all_online_edges.shape)

    useful_nodes = torch.unique(all_online_edges[:,1])
    print("num useful nodes = ",useful_nodes.shape)

    useful_edges = []
    for e in dataset['init_edge_index'].T:
        if e[0].item() in useful_nodes and e[1].item() in useful_nodes:
            useful_edges.append(tuplify(e))

    nodes_made_by_ue = []
    for e in useful_edges:
        e0 = int(e[0])
        e1 = int(e[1])
        if e0 not in nodes_made_by_ue:
            nodes_made_by_ue.append(e0)
        if e1 not in nodes_made_by_ue:
            nodes_made_by_ue.append(e1)
    print("num nodes in useful edge=",len(nodes_made_by_ue))

    extra_nodes = []
    for n in useful_nodes:
        if n not in nodes_made_by_ue:
            extra_nodes.append(n.item())
    print("extra nodes: ",extra_nodes)

    base_graph_edges = useful_edges
    base_extra_nodes = extra_nodes

    figsize = (30,30)
    fig_dir = './figs'
    os.makedir(fig_dir,exists_ok=True)
    gif_name = os.path.join(fig_dir,'online_link_pred_75.gif')
    visualize(base_graph_edges, base_extra_nodes, new_nodes, new_given_edges,pred_new_node_edges, correct_new_edge, figsize,gif_name,framerate=12)