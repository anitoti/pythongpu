import kuramoto
import gnm_random_graph
from .kuramoto import KuramotoSimulator
import networkx as nx

adj = nx.to_numpy_array(nx.gnm_random_graph(100, 500))
sim = KuramotoSimulator(adj, K=2.0, device="cuda")
history = sim.simulate(steps=1000)