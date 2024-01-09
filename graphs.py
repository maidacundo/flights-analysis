import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from typing import Dict, List, Tuple

def plot_spatial_network_centrality_distribution(centralities: Dict[str, float], metric: str):
    """Plot the centrality distibution and the cumulative centrality distribution in the spatial network.

    Parameters
    ----------
    centralities : Dict[str, float]
        Dictionary that for each node (key) contains its centrality (value)
    metric : str
        The name of the centrality metric.
    """
    hist, bin_edges = np.histogram(list(centralities.values()), bins=20, density=False, range=(0., 1.))

    width = 0.7 * (bin_edges[1] - bin_edges[0])
    center = (bin_edges[:-1] + bin_edges[1:]) / 2

    cmap = plt.get_cmap('tab10')

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)  
    plt.xticks(bin_edges[::2])
    plt.bar(center, hist, width=width, color=cmap.colors[0])
    plt.xlabel('centrality')
    plt.title(f'{metric.capitalize()} distribution for the spatial network ')
    plt.grid(axis='y')

    hist, bin_edges = np.histogram(list(centralities.values()), bins=20, density=True, range=(0., 1.))
    cumsum = np.cumsum(hist) * (bin_edges[1] - bin_edges[0])
    center = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Cumulative distribution
    plt.subplot(1, 2, 2) 
    plt.xticks(bin_edges[::2])
    plt.plot(center, cumsum, 'o-')
    plt.title(f'{metric.capitalize()} cumulative distribution for the spatial network')
    plt.xlabel('centrality')
    plt.grid(axis='y')
    plt.show()