import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

graph_path = os.getcwd()[:-7] + "data/graph_data/"


def create_overlap(G, overlap):

    if overlap == 1.0:
        return G.edges()

    G_original = G.copy()
    num_edges = len(G_original.edges())

    num_rewire = 0
    G_temp = G_original.copy()
    G_temp = nx.double_edge_swap(G_temp, nswap=num_rewire, max_tries=num_rewire*10)
    G_inter = nx.intersection(G_temp, G_original)

    while np.abs((len(G_inter.edges())/num_edges) - overlap) > 0.02:
        
        num_rewire += int(num_edges*0.005)
        
        G_temp = G_original.copy()
        G_temp = nx.double_edge_swap(G_temp, nswap=num_rewire, max_tries=num_rewire*10)
        
        G_inter = nx.intersection(G_temp, G_original)

        if num_rewire > num_edges*2:
            break

    if np.abs((len(G_inter.edges())/num_edges) - overlap) > 0.02:
        print('could not find a suitable overlap')
        return None
    else:
        min_overlap = len(G_inter.edges())/num_edges

        for i in range(50):

            G_new_temp = G_original.copy()
            G_new_temp = nx.double_edge_swap(G_new_temp, nswap=num_rewire, max_tries=num_rewire*10)
            G_inter = nx.intersection(G_new_temp, G_original)

            if len(G_inter.edges())/num_edges < min_overlap:
                min_overlap = len(G_inter.edges())/num_edges
                G_temp = G_new_temp

        return G_temp.edges()



def create_network(N, num_links, num_triangles, overlap):

    num_ind_links = num_links - 2*num_triangles

    deg_seq = [(num_ind_links, num_triangles) for i in range(N)]

    G = nx.random_clustered_graph(deg_seq, create_using=nx.Graph)
    G = nx.Graph(G)
    G.remove_edges_from(nx.selfloop_edges(G))

    degrees = [G.degree(n) for n in G.nodes()]

    tri_counts = nx.triangles(G)
    triangles = [tri_counts[n] for n in G.nodes()]

    count = 0

    while (len(np.unique(degrees)) != 1) or (len(np.unique(triangles)) != 1) or (np.unique(triangles)[0] != num_triangles) or (np.unique(degrees)[0] != num_links):

        G = nx.random_clustered_graph(deg_seq, create_using=nx.Graph)
        G = nx.Graph(G)
        G.remove_edges_from(nx.selfloop_edges(G))

        degrees = [G.degree(n) for n in G.nodes()]

        tri_counts = nx.triangles(G)

        triangles = [tri_counts[n] for n in G.nodes()]

        count += 1

        # if count % 100 == 0:
        #     print(count)

        if count > 1000:
            print('could not find a suitable network')
            break


    if count <= 1000:
        pair_dat = [[0,0,0,0] for i in range(len(G.nodes()))]
        higher_order_dat = [[0,0,0,0] for i in range(len(G.nodes()))]

        all_cliques = nx.enumerate_all_cliques(G)
        all_triangles = [c for c in all_cliques if len(c) == 3]

        for this_triangle in all_triangles:
            for i, node in enumerate(this_triangle):
                node1, node2 = this_triangle[i-2] + 1, this_triangle[i-1] + 1
                node += 1
                if higher_order_dat[node-1][0] == 0:
                    higher_order_dat[node-1][0] = node1
                    higher_order_dat[node-1][1] = node2
                else:
                    higher_order_dat[node-1][2] = node1
                    higher_order_dat[node-1][3] = node2

        
        ''' create overlap '''
        all_edges = create_overlap(G, overlap)   
        if all_edges is None:
            return None, None     
        ''' created overlap '''


        for this_edge in all_edges:
            node1 = this_edge[0] + 1
            node2 = this_edge[1] + 1

            # Fill the first zero slot for node1
            for i in range(4):
                if pair_dat[node1 - 1][i] == 0:
                    pair_dat[node1 - 1][i] = node2
                    break

            # Fill the first zero slot for node2
            for i in range(4):
                if pair_dat[node2 - 1][i] == 0:
                    pair_dat[node2 - 1][i] = node1
                    break


        return np.array(pair_dat), np.array(higher_order_dat)
    
    else:
        return None, None
    


num_graphs = 100
N = 1500
num_links = int(sys.argv[1])
num_triangles = int(sys.argv[2])
overlap = float(sys.argv[3])

overlap_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

for overlap in overlap_list:

    count = num_graphs
    for i in range(num_graphs):
        pair, high = create_network(N = N, num_links = num_links, num_triangles = num_triangles, overlap = overlap)

        if pair is None or high is  None:
            count -= 1
            continue

        else:
            np.savetxt(graph_path + f'pairwise_graph_N_{N}_kp_{num_links}_kh_{num_triangles}_o_{overlap}_{i}.txt', np.transpose(pair), fmt='%d')
            np.savetxt(graph_path + f'higher_order_graph_N_{N}_kp_{num_links}_kh_{num_triangles}_o_{overlap}_{i}.txt', np.transpose(high), fmt='%d')

    print(f'created {count} graphs for overlap={overlap}')

