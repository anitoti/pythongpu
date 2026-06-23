import networkx as nx
import torch
import numpy as np
from .utils import get_laplacian
import matplotlib # this and Agg dont open a new window in the server, so just a file saves
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def generate_gnm_laplacian(n, m, norm=None, device="cpu"):
    g = nx.gnm_random_graph(n, m)
    adj = nx.to_numpy_array(g) 
    
    # uses the pytorch utils.py we made earlier 
    L = get_laplacian(adj, norm=norm, device=device) 

    # generate image of nodes and edges per run
    nx.draw(g, node_size=20)
    plt.savefig('network.png')
    plt.clf()
    plt.close()

    # calculate probability 
    p = m / (n * (n - 1) / 2)
    print("probability = ",p)   

    return L

if __name__ == "__main__":
    generate_gnm_laplacian(100, 500)

# to run, 
# python3 -m src.pythongpu.gnm_random_graph