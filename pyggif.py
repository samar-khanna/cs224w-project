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

def visualize(base_graph_edges,new_nodes,new_given_edges,pred_new_node_edges,correct_new_edge,figsize, file_path):
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

    # Base Graph
    G_old = nx.Graph()
    G_old.add_edges_from(base_graph_edges)
    plt.figure(figsize=figsize)
    plt.xlim(min_x_pos,max_x_pos)
    plt.ylim(min_y_pos,max_y_pos)
    pos_old = {i:pos[i] for i in G_old.nodes()}
    node_labels_old = {i:i for i in G_old.nodes()}
    node_color_old = ['b' for node in G_old.nodes()]

    nx.draw(G_old, pos=pos_old,node_color=node_color_old, labels=node_labels_old, font_color='white')
    filename = os.path.join(file_path,f'{0}.png')
    filenames.append(filename)
    plt.savefig(filename)

    G_pred = nx.Graph()
    G_pred.clear()
    G_pred.add_edges_from(base_graph_edges)
    edge_color = {}

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
                elif rev_edge in pred_new_node_edges[index] and rev_edge in correct_new_edge[index]:
                    edge_color[edge] = 'purple'
                    edges_learnt.append(rev_edge)
                else:
                    edge_color[edge] = 'white'
                    # print('whiten',edge)
            elif rev_edge in new_given_edges[index]:
                edge_color[edge] = 'blue'
                edges_learnt.append(rev_edge)
                # print("given ",edge)
            elif rev_edge in correct_new_edge[index] and rev_edge in pred_new_node_edges[index] :
                edge_color[edge] = 'green'
                edges_learnt.append(rev_edge)
                # print(edge, "is correct")
            elif rev_edge in pred_new_node_edges[index]:
                edge_color[edge] = 'red'
                # print(edge, "is wrong")
            elif rev_edge in correct_new_edge[index]:
                edge_color[edge] = 'yellow'
                # print(edge, "was missed")
            # print('edges learnt so far')
            # print(edges_learnt)
            # print('-------')
        node_color = ['r' if node==new_node else 'b' for node in G_pred.nodes()]
        ec = collections.OrderedDict(sorted(edge_color.items()))
        plt.figure(figsize=figsize)
        plt.xlim(min_x_pos,max_x_pos)
        plt.ylim(min_y_pos,max_y_pos)
        nx.draw(G_pred, pos=pos_new, labels=node_labels, edge_color=ec.values(), node_color=node_color, font_color='white')
        filename = os.path.join(file_path,f'{index+1}.png')
        filenames.append(filename)
        plt.savefig(filename)

    # make gif
    with imageio.get_writer(os.path.join(file_path,'online_link_pred.gif'), mode='I') as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)

    # remove frames
    for filename in set(filenames):
        os.remove(filename)

if __name__ == "__main__":
    base_graph_edges = [(1,2),(1,3),(2,3),(2,4),(3,4)]
    new_nodes = [5,6,7]
    new_given_edges = [[(5,2)],[(6,4)],[]]
    pred_new_node_edges = [[(5,1),(5,4)],[(6,2),(6,5)],[(7,1),(7,5)]]
    correct_new_edge = [[(5,1),(5,3)],[(6,2),(6,5)],[(7,2)]]

    figsize = (5,5)
    file_path = './figs'
    os.makedirs(file_path, exist_ok=True)
    visualize(base_graph_edges,new_nodes,new_given_edges,pred_new_node_edges,correct_new_edge,figsize, file_path)