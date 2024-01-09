import numpy as np
import networkx as nx
from metrics import get_nodes_betweenness_centrality

def percentile_graph(scores, passengers, num=10):
    limits = np.linspace(0, 100, num=num+1)
    percentile_scores = []
    for i in range(len(limits) - 1):
        start = limits[i]
        end = limits[i + 1]
        relevant_scores = [scores[k] for k in passengers if passengers[k] > start and passengers[k] <= end]
        percentile_scores.append(np.mean(relevant_scores))
    return percentile_scores

def compute_average(scores, weights=None):
    if weights is None:
        return sum([scores[k] for k in scores.keys()]) / len(scores)
    relevant_weights = [weights[k] for k in scores.keys()]
    if sum(relevant_weights) == 0:
        return 0
    else:
        return sum([scores[k] * weights[k] for k in scores.keys()]) / sum(relevant_weights)
    
def iata_to_name(airports, iata):
    return airports[iata]['name']


def remove_weakly_connected_components(network):
    components = list(nx.strongly_connected_components(network))
    c = sorted(components, key=len, reverse=True)[0]
    subgraph = network.subgraph(c)
    return subgraph

def remove_airport(network, iata, betweenness_before, pagerank_before):
    new_network = nx.DiGraph(network)
    new_network.remove_node(iata)
    num_connected_comps = nx.number_connected_components(new_network.to_undirected())
    betweenness_after = get_nodes_betweenness_centrality(new_network, weight='ramp_to_ramp')
    pagerank_after = nx.pagerank(new_network, weight='passengers')

    max_variation_betweenness = 0
    max_variation_node_betweenness = None
    for node in betweenness_after:
        variation = betweenness_after[node] - betweenness_before[node]
        if variation > max_variation_betweenness:
            max_variation_betweenness = variation
            max_variation_node_betweenness = node
    
    max_variation_pagerank = 0
    max_variation_node_pagerank = None
    for node in pagerank_after:
        variation = pagerank_after[node] - pagerank_before[node]
        if variation > max_variation_pagerank:
            max_variation_pagerank = variation
            max_variation_node_pagerank = node

    new_network = remove_weakly_connected_components(new_network)
    avg_short_path = nx.average_shortest_path_length(new_network, weight='ramp_to_ramp')
    diameter = nx.diameter(new_network)
    return avg_short_path, diameter, num_connected_comps, max_variation_betweenness, max_variation_node_betweenness, max_variation_pagerank, max_variation_node_pagerank