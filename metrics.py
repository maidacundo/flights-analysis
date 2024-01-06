"""
MIT License

Copyright (c) 2020 Shuaib

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import networkx as nx

def weighted_hits(G, weight='weight', max_iter=1_000, tol=1.0e-8, nstart=None, normalized=True):
    """Returns HITS hubs and authorities values for nodes.
    The HITS algorithm computes two numbers for a node.
    Authorities estimates the node value based on the incoming links.
    Hubs estimates the node value based on outgoing links.
    Parameters
    ----------
    G : graph
      A NetworkX graph
    max_iter : integer, optional
      Maximum number of iterations in power method.
    tol : float, optional
      Error tolerance used to check convergence in power method iteration.
    nstart : dictionary, optional
      Starting value of each node for power method iteration.
    normalized : bool (default=True)
       Normalize results by the sum of all of the values.
    Returns
    -------
    (hubs,authorities) : two-tuple of dictionaries
       Two dictionaries keyed by node containing the hub and authority
       values.
    Raises
    ------
    PowerIterationFailedConvergence
        If the algorithm fails to converge to the specified tolerance
        within the specified number of iterations of the power iteration
        method.
    Examples
    --------
    >>> G = nx.path_graph(4)
    >>> h, a = nx.hits(G)
    Notes
    -----
    The eigenvector calculation is done by the power iteration method
    and has no guarantee of convergence.  The iteration will stop
    after max_iter iterations or an error tolerance of
    number_of_nodes(G)*tol has been reached.
    The HITS algorithm was designed for directed graphs but this
    algorithm does not check if the input graph is directed and will
    execute on undirected graphs.
    References
    ----------
    .. [1] A. Langville and C. Meyer,
       "A survey of eigenvector methods of web information retrieval."
       http://citeseer.ist.psu.edu/713792.html
    .. [2] Jon Kleinberg,
       Authoritative sources in a hyperlinked environment
       Journal of the ACM 46 (5): 604-32, 1999.
       doi:10.1145/324133.324140.
       http://www.cs.cornell.edu/home/kleinber/auth.pdf.
    """
    if type(G) == nx.MultiGraph or type(G) == nx.MultiDiGraph:
        raise Exception("hits() not defined for graphs with multiedges.")
    if len(G) == 0:
        return {}, {}
    # choose fixed starting vector if not given
    if nstart is None:
        h = dict.fromkeys(G, 1.0 / G.number_of_nodes())
    else:
        h = nstart
        # normalize starting vector
        s = 1.0 / sum(h.values())
        for k in h:
            h[k] *= s
    for _ in range(max_iter):  # power iteration: make up to max_iter iterations
        hlast = h
        h = dict.fromkeys(hlast.keys(), 0)
        a = dict.fromkeys(hlast.keys(), 0)
        # this "matrix multiply" looks odd because it is
        # doing a left multiply a^T=hlast^T*G
        for n in h:
            for nbr in G[n]:
                a[nbr] += hlast[n] * G[n][nbr][weight]
        # now multiply h=Ga
        for n in h:
            for nbr in G[n]:
                h[n] += a[nbr] * G[n][nbr][weight]
        # normalize vector
        s = 1.0 / max(h.values())
        for n in h:
            h[n] *= s
        # normalize vector
        s = 1.0 / max(a.values())
        for n in a:
            a[n] *= s
        # check convergence, l1 norm
        err = sum([abs(h[n] - hlast[n]) for n in h])
        if err < tol:
            break
    else:
        raise nx.PowerIterationFailedConvergence(max_iter)
    if normalized:
        s = 1.0 / sum(a.values())
        for n in a:
            a[n] *= s
        s = 1.0 / sum(h.values())
        for n in h:
            h[n] *= s
    return h, a


"""
Edited from the Networkx Documentation

Find the k-cores of a graph.

The k-core is found by recursively pruning nodes with degrees less than k.

See the following references for details:

An O(m) Algorithm for Cores Decomposition of Networks
Vladimir Batagelj and Matjaz Zaversnik, 2003.
https://arxiv.org/abs/cs.DS/0310049

Generalized Cores
Vladimir Batagelj and Matjaz Zaversnik, 2002.
https://arxiv.org/pdf/cs/0202039

For directed graphs a more general notion is that of D-cores which
looks at (k, l) restrictions on (in, out) degree. The (k, k) D-core
is the k-core.

D-cores: Measuring Collaboration of Directed Graphs Based on Degeneracy
Christos Giatsidis, Dimitrios M. Thilikos, Michalis Vazirgiannis, ICDM 2011.
http://www.graphdegeneracy.org/dcores_ICDM_2011.pdf

Multi-scale structure and topological anomaly detection via a new network \
statistic: The onion decomposition
L. Hébert-Dufresne, J. A. Grochow, and A. Allard
Scientific Reports 6, 31708 (2016)
http://doi.org/10.1038/srep31708

Edits list:
- Addition of types
- Addition of extra parameters or controls in some functions to allow the use of weights
- Function `core_number_weighted` created from scratch.

"""
import networkx as nx
from networkx.exception import NetworkXError
from networkx.utils import not_implemented_for
from typing import Callable, Dict, Optional

def core_number_weighted(network: nx.Graph, weight: str) -> Dict[str, float]:
    """Get the weighted core number of each node in the network

    Parameters
    ----------
    network : Graph
        The mnetwork from which the core numbers are obtained
    weight : str
        The name of the weight on the edges that has to be used to compute the weighted core number of the nodes

    Returns
    -------
    { str: float }
        Dictionary containing for each node its weighted core number
    """
    # Get weighted node degree dictionary.
    degrees = dict(network.degree(weight=weight))
    # Sort nodes by non-decreasing degree.
    nodes = sorted(degrees, key=degrees.get)

    # Initialize core_number dictionary
    cores = {k: 0 for k in nodes}

    for i in range(len(nodes)):
        # Get current node
        u = nodes[i]
        # Initialize its core value as its degree
        cores[u] = degrees[u]
        # Update neighbouring nodes core number
        for w in list(nx.all_neighbors(network, u)):
            if cores[u] < degrees[w]:
                degrees[w] = max(degrees[w] - network[u][w][weight], cores[u])
        nodes[i+1:] = sorted({k: v for k, v in degrees.items() if k in nodes[i+1:]}, key=degrees.get)

    return cores


#@nx._dispatch
@not_implemented_for("multigraph")
def core_number(G: nx.Graph):
    """Returns the core number for each vertex.

    A k-core is a maximal subgraph that contains nodes of degree k or more.

    The core number of a node is the largest value k of a k-core containing
    that node.

    Parameters
    ----------
    G : NetworkX graph
       A graph or directed graph

    Returns
    -------
    core_number : dictionary
       A dictionary keyed by node to the core number.

    Raises
    ------
    NetworkXError
        The k-core is not implemented for graphs with self loops
        or parallel edges.

    Notes
    -----
    Not implemented for graphs with parallel edges or self loops.

    For directed graphs the node degree is defined to be the
    in-degree + out-degree.

    References
    ----------
    .. [1] An O(m) Algorithm for Cores Decomposition of Networks
       Vladimir Batagelj and Matjaz Zaversnik, 2003.
       https://arxiv.org/abs/cs.DS/0310049
    """
    if nx.number_of_selfloops(G) > 0:
        msg = (
            "Input graph has self loops which is not permitted; "
            "Consider using G.remove_edges_from(nx.selfloop_edges(G))."
        )
        raise NetworkXError(msg)
    degrees = dict(G.degree())
    # Sort nodes by degree.
    nodes = sorted(degrees, key=degrees.get)
    bin_boundaries = [0]
    curr_degree = 0
    for i, v in enumerate(nodes):
        if degrees[v] > curr_degree:
            bin_boundaries.extend([i] * (degrees[v] - curr_degree))
            curr_degree = degrees[v]
    node_pos = {v: pos for pos, v in enumerate(nodes)}
    # The initial guess for the core number of a node is its degree.
    core = degrees
    nbrs = {v: list(nx.all_neighbors(G, v)) for v in G}
    for v in nodes:
        for u in nbrs[v]:
            if core[u] > core[v]:
                nbrs[u].remove(v)
                pos = node_pos[u]
                bin_start = bin_boundaries[core[u]]
                node_pos[u] = bin_start
                node_pos[nodes[bin_start]] = pos
                nodes[bin_start], nodes[pos] = nodes[pos], nodes[bin_start]
                bin_boundaries[core[u]] += 1
                core[u] -= 1
    return core



def _core_subgraph(G: nx.Graph, k_filter: Callable[[float, float, Dict[str, float]],bool], k: Optional[float] = None,
                   core: Optional[Dict[str, float]] = None, weight: Optional[str] = None) -> nx.Graph:
    """Returns the subgraph induced by nodes passing filter `k_filter`.

    Parameters
    ----------
    G : NetworkX graph
       The graph or directed graph to process
    k_filter : filter function
       This function filters the nodes chosen. It takes three inputs:
       A node of G, the filter's cutoff, and the core dict of the graph.
       The function should return a Boolean value.
    k : int, optional
      The order of the core. If not specified use the max core number.
      This value is used as the cutoff for the filter.
    core : dict, optional
      Precomputed core numbers keyed by node for the graph `G`.
      If not specified, the core numbers will be computed from `G`.
    weight : str, optional
      The name of the weight on the edges that has to be used to compute the weighted core number of the nodes
    """
    if core is None:
        if weight is None:
          core = core_number(G)
        else:
          core = core_number_weighted(G, weight)
    if k is None:
        k = sum(core.values()) / len(core)
    nodes = (v for v in core if k_filter(v, k, core))
    return G.subgraph(nodes).copy()

def k_core(G: nx.Graph, k: Optional[float] =None, core_number: Optional[Optional[Dict[str, float]]] = None,
           weight: Optional[str] = None) -> nx.Graph:
    """Returns the k-core of G.

    A k-core is a maximal subgraph that contains nodes of degree k or more.

    Parameters
    ----------
    G : NetworkX graph
      A graph or directed graph
    k : int, optional
      The order of the core.  If not specified return the main core.
    core_number : dictionary, optional
      Precomputed core numbers for the graph G.
    weight : str, optional
      The weight used to compute the k-core

    Returns
    -------
    G : NetworkX graph
      The k-core subgraph

    Raises
    ------
    NetworkXError
      The k-core is not defined for graphs with self loops or parallel edges.

    Notes
    -----
    The main core is the core with the largest degree.

    Not implemented for graphs with parallel edges or self loops.

    For directed graphs the node degree is defined to be the
    in-degree + out-degree.

    Graph, node, and edge attributes are copied to the subgraph.

    See Also
    --------
    core_number

    References
    ----------
    .. [1] An O(m) Algorithm for Cores Decomposition of Networks
       Vladimir Batagelj and Matjaz Zaversnik,  2003.
       https://arxiv.org/abs/cs.DS/0310049
    """

    def k_filter(v, k, c):
        return c[v] >= k

    return _core_subgraph(G, k_filter, k, core_number, weight)

from copy import deepcopy
import networkx as nx
from statistics import geometric_mean
from typing import Dict, Optional, Tuple

def _normalize_metric(metric_dict: Dict[str, float]) -> Dict[str, float]:
    """Function to normalize the metric across the whole dictionary according to min-max scale.

    Parameters
    ----------
    metric_dict : { str: float }
        Dictionary to normalize where the keys are nodes and the values the relative metrics results.

    Returns
    -------
    { str: float }
        The normalized dictionary.
    """
    return { n: (v - min(metric_dict.values())) / (max(metric_dict.values()) - min(metric_dict.values()))
                for n, v in metric_dict.items() }

def get_nodes_degree_centrality(network: nx.Graph, normalize: bool = True,
                                   weight: Optional[str] = None) -> Dict[str, float]:
    """Get the in-degree centrality of all the nodes in the network.

    Parameters
    ----------
    network : Graph
        The network for which the centrality of the nodes is computed.
    normalize : bool, optional
        Whether to normalize or not the centrality results by min-max scale, by default True
    weight : str, optional
        The edge weight to use to compute the centrality measures, by default None

    Returns
    -------
    { str: float }
        Dictionary where the keys are nodes and the values the relative centrality values.
    """
    metric_dict = { n: network.degree(n, weight=weight) for n in network.nodes() }
    return metric_dict if not normalize else _normalize_metric(metric_dict)

def get_nodes_in_degree_centrality(network: nx.Graph, normalize: bool = True,
                                   weight: Optional[str] = None) -> Dict[str, float]:
    """Get the in-degree centrality of all the nodes in the network.

    Parameters
    ----------
    network : Graph
        The network for which the centrality of the nodes is computed.
    normalize : bool, optional
        Whether to normalize or not the centrality results by min-max scale, by default True
    weight : str, optional
        The edge weight to use to compute the centrality measures, by default None

    Returns
    -------
    { str: float }
        Dictionary where the keys are nodes and the values the relative centrality values.
    """
    metric_dict = { n: network.in_degree(n, weight=weight) for n in network.nodes() }
    return metric_dict if not normalize else _normalize_metric(metric_dict)

def get_nodes_out_degree_centrality(network: nx.Graph, normalize: bool = True,
                                    weight: Optional[str] = None) -> Dict[str, float]:
    """Get the out-degree centrality of all the nodes in the network.

    Parameters
    ----------
    network : Graph
        The network for which the centrality of the nodes is computed.
    normalize : bool, optional
        Whether to normalize or not the centrality results by min-max scale, by default True
    weight : str, optional
        The edge weight to use to compute the centrality measures, by default None

    Returns
    -------
    { str: float }
        Dictionary where the keys are nodes and the values the relative centrality values.
    """
    metric_dict = { n: network.out_degree(n, weight=weight) for n in network.nodes() }
    return metric_dict if not normalize else _normalize_metric(metric_dict)

def get_nodes_betweenness_centrality(network: nx.Graph, normalize: bool = True, weight: Optional[str] = None,
                                     seed: int = 42) -> Dict[str, float]:
    """Get the betweenness centrality of all the nodes in the network.

    Parameters
    ----------
    network : Graph
        The network for which the centrality of the nodes is computed.
    normalize : bool, optional
        Whether to normalize or not the centrality results by min-max scale, by default True.
    weight : str, optional
        The edge weight to use to compute the centrality measures, by default None.
    seed : int, optional
        The seed to use, by default 42.

    Returns
    -------
    { str: float }
        Dictionary where the keys are nodes and the values the relative centrality values.
    """
    metric_dict = nx.betweenness_centrality(network, k=None, normalized=False, weight=weight, endpoints=False, seed=seed)
    return metric_dict if not normalize else _normalize_metric(metric_dict)

def get_nodes_closeness_centrality(network: nx.Graph, normalize: bool = True,
                                   weight: Optional[str] = None) -> Dict[str, float]:
    """Get the closeness centrality of all the nodes in the network.

    Parameters
    ----------
    network : Graph
        The network for which the centrality of the nodes is computed.
    normalize : bool, optional
        Whether to normalize or not the centrality results by min-max scale, by default True
    weight : str, optional
        The edge weight to use to compute the centrality measures, by default None

    Returns
    -------
    { str: float }
        Dictionary where the keys are nodes and the values the relative centrality values.
    """
    metric_dict = nx.closeness_centrality(network, u=None, distance=weight)
    return metric_dict if not normalize else _normalize_metric(metric_dict)

def get_nodes_pagerank_centrality(network: nx.Graph, normalize: bool = True,
                                  weight: Optional[str] = None) -> Dict[str, float]:
    """Get the PageRank centrality of all the nodes in the network.

    Parameters
    ----------
    network : Graph
        The network for which the centrality of the nodes is computed.
    normalize : bool, optional
        Whether to normalize or not the centrality results by min-max scale, by default True
    weight : str, optional
        The edge weight to use to compute the centrality measures, by default None

    Returns
    -------
    { str: float }
        Dictionary where the keys are nodes and the values the relative centrality values.
    """
    metric_dict = nx.pagerank(network, alpha=0.85, max_iter=100, tol=1e-06, nstart=None, weight=weight, dangling=None)
    return metric_dict if not normalize else _normalize_metric(metric_dict)

def get_average_shortest_path(network: nx.Graph, normalize: bool = True, weight: Optional[str] = None) -> Dict[str, float]:
    """Get the average shortest path of the network.

    Parameters
    ----------
    network : Graph
        The network for which the average shortest path is computed.
    weight : str, optional
        The edge weight to use to compute the average shortest path, by default None

    Returns
    -------
    float
        The average shortest path.
    """
    metric_dict = nx.closeness_centrality(network, u=None, distance=weight)
    return metric_dict if not normalize else _normalize_metric(metric_dict)

def get_nodes_hits_centrality(network: nx.Graph, normalize: bool = True,
                              weight: Optional[str] = None) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Get the HITS centrality (Hubs and Authorities) of all the nodes in the network.

    Parameters
    ----------
    network : Graph
        The network for which the centrality of the nodes is computed.
    normalize : bool, optional
        Whether to normalize or not the centrality results by min-max scale, by default True
    weight : str, optional
        The edge weight to use to compute the centrality measures, by default None

    Returns
    -------
    { str: float }
        Dictionary where the keys are nodes and the values the relative centrality values.
    """
    # metric_dict = weighted_hits(network, normalized=normalize, weight=weight)
    hubs_dict, auth_dict = nx.hits(network)
    return (hubs_dict, auth_dict) if not normalize else (_normalize_metric(hubs_dict), _normalize_metric(auth_dict))

def normalize_centrality_measures(centrality_dict: Dict[int, Dict[str, float]]) -> Dict[int, Dict[str, float]]:
    """Function to normalize with min-max scale the centrality measures across a dictionary of dictionaries of metrics results

    Parameters
    ----------
    centrality_dict : { int : { str : float } }
        The dictionary of dictionaries of metrics.

    Returns
    -------
    { int : { str : float } }
        The normalized dictionary
    """
    centrality_dict = deepcopy(centrality_dict)

    values_list = [v for centralities in centrality_dict.values() for v in centralities.values()]
    min_value = min(values_list)
    max_value = max(values_list)

    for centralities in centrality_dict.values():
        for k, v in centralities.items():
            centralities[k] = (v - min_value) / (max_value - min_value)

    return centrality_dict

def get_girvan_newman_communities(network: nx.Graph, weight: Optional[str] = None, k: int = 2,
                                  seed: int = 42) -> Dict[str, int]:
    """Function to get the Girvan-Newmann partition from a network.

    Parameters
    ----------
    network : Graph
        The network from which the communities are obtained.
    weight : str, optional
        The edge weight to use to compute the communities, by default None.
    k : int, optional
        How many communities to find, by default 2
    seed : int, optional
        The seed to use, by default 42

    Returns
    -------
    { str: int }
        Dictionary describing for each node (key) the relative community (value).
    """
    def most_central_edge(network: nx.Graph):
        centrality = nx.edge_betweenness_centrality(network, weight=weight, seed=seed)
        return max(centrality, key=centrality.get)

    communities_iterator = nx.community.girvan_newman(network, most_valuable_edge=most_central_edge)

    for _ in range(k - 1):
        communities = next(communities_iterator)

    return { c: i for i, community in enumerate(communities) for c in community }

def get_k_cores_communities(network: nx.Graph, weight: Optional[str] = None, k: Optional[int] = None) -> Dict[str, int]:
    """Function to get the k-core partition from a network.

    Parameters
    ----------
    network : Graph
        The network from which the communities are obtained.
    weight : str, optional
        The edge weight to use to compute the communities, by default None.
    k : int, optional
        The core number that the nodes must reach to be part od a community, by default None.
        If None, it will be initialized as the geometric mean of all edges of the network.

    Returns
    -------
    { str: int }
        Dictionary describing for each node (key) the relative community (value).
    """
    new_network = deepcopy(network)
    node_cores_dict = {}
    n = 0

    while len(new_network.nodes()):
        try:
            k_core_subgraph = k_core(new_network, k=k, weight=weight)
        except ValueError:
            break

        for node in k_core_subgraph.nodes():
            node_cores_dict[node] = n
            new_network.remove_node(node)

        n += 1

    for node in new_network.nodes():
        node_cores_dict[node] = n

    return node_cores_dict

def get_clique_percolation_communities(network: nx.Graph, k: Optional[int] = 2,
                                       weight: Optional[str] = None) -> Dict[str, int]:
    """Function to get the Clique Percolation partition from a network.

    Parameters
    ----------
    network : Graph
        The network from which the communities are obtained.
    weight : str, optional
        The edge weight to use to compute the communities, by default None.
    k : int, optional
        The minimum clique size to consider, by default 2.

    Returns
    -------
    { str: int }
        Dictionary describing for each node (key) the relative community (value).
    """
    new_network = deepcopy(network)
    n = 0
    communities = dict()

    while len(new_network.nodes()):
        try:
            l = geometric_mean([d[weight] for _, _, d in new_network.edges(data=True)])
        except:
            l = 0

        community_iterator = k_clique_communities(new_network, weight=weight, l=l, k=k)
        all_cliques = list(community_iterator)
        if len(all_cliques) == 0:
            break
        for clique in all_cliques:
            for c in clique:
                communities[c] = n
                if new_network.has_node(c):
                    new_network.remove_node(c)
            n += 1

    for node in new_network.nodes():
        communities[node] = n

    return communities

def get_louvain_communities(network: nx.Graph, weight: Optional[str] = None) -> Dict[str, int]:
    """Function to get the Louvain partition from a network.

    Parameters
    ----------
    network : Graph
        The network from which the communities are obtained.
    weight : str, optional
        The edge weight to use to compute the communities, by default None.

    Returns
    -------
    { str: int }
        Dictionary describing for each node (key) the relative community (value).
    """
    communities = nx.community.louvain_communities(network, weight=weight, resolution=1, threshold=1e-07, seed=42)
    return { c: i for i, community in enumerate(communities) for c in community }

def get_modularity_score(network: nx.Graph, node_community_dict: Dict[str, int], weight: str) -> float:
    """Function to get the modularity score of a network partition.

    Parameters
    ----------
    network : Graph
        The network partition
    node_community_dict : { str: int }
        Dictionary describing for each node (key) the relative partition (value).
    weight : str
        The edge weight to use to compute the odularity score.

    Returns
    -------
    float
        The modularity score.
    """
    communities_labels = set(node_community_dict.values())
    communities = {l: [] for l in communities_labels}
    for k, v in node_community_dict.items():
        communities[v].append(k)
    return nx.community.modularity(network, communities.values(), weight=weight)




"""
Edited from the Networkx Documentation

Edits list:
- Addition of types
- Addition of extra parameters or controls in some functions to allow the use of weights
"""

from collections import defaultdict
from statistics import geometric_mean
import networkx as nx

def k_clique_communities(G: nx.Graph, weight: str, k: int = 2, l: float = 100):
    """Find k-clique communities in graph using the percolation method.

    A k-clique community is the union of all cliques of size k that
    can be reached through adjacent (sharing k-1 nodes) k-cliques.

    Parameters
    ----------
    G : NetworkX graph

    weigth : str
        Weight to use to compute k-clique

    k : int
       Size of smallest clique

    cliques: list or generator
       Precomputed cliques (use networkx.find_cliques(G))

    Returns
    -------
    Yields sets of nodes, one for each k-clique community.

    Examples
    --------
    >>> from networkx.algorithms.community import k_clique_communities
    >>> G = nx.complete_graph(5)
    >>> K5 = nx.convert_node_labels_to_integers(G, first_label=2)
    >>> G.add_edges_from(K5.edges())
    >>> c = list(k_clique_communities(G, 4))
    >>> sorted(list(c[0]))
    [0, 1, 2, 3, 4, 5, 6]
    >>> list(k_clique_communities(G, 6))
    []

    References
    ----------
    .. [1] Gergely Palla, Imre Derényi, Illés Farkas1, and Tamás Vicsek,
       Uncovering the overlapping community structure of complex networks
       in nature and society Nature 435, 814-818, 2005,
       doi:10.1038/nature03607
    """
    if k < 2:
        raise nx.NetworkXError(f"k={k}, k must be greater than 1.")

    cliques_ = nx.find_cliques(G)

    cliques = []

    for c in cliques_:
        subgraph = G.subgraph(c)
        try:
            gm = geometric_mean([w[weight] for _, _, w in subgraph.edges(data=True)])
        except:
            gm = 0
        if len(c) >= k and gm >= l:
            cliques.append(frozenset(c))

    # cliques = [frozenset(c) for c in cliques if len(c) >= k (multiply_values)**(1/n)]

    # First index which nodes are in which cliques
    membership_dict = defaultdict(list)
    for clique in cliques:
        for node in clique:
            membership_dict[node].append(clique)

    # For each clique, see which adjacent cliques percolate
    perc_graph = nx.Graph()
    perc_graph.add_nodes_from(cliques)
    for clique in cliques:
        for adj_clique in _get_adjacent_cliques(clique, membership_dict):
            if len(clique.intersection(adj_clique)) >= (k - 1):
                perc_graph.add_edge(clique, adj_clique)

    # Connected components of clique graph with perc edges
    # are the percolated cliques
    for component in nx.connected_components(perc_graph):
        yield (frozenset.union(*component))



def _get_adjacent_cliques(clique, membership_dict):
    adjacent_cliques = set()
    for n in clique:
        for adj_clique in membership_dict[n]:
            if clique != adj_clique:
                adjacent_cliques.add(adj_clique)
    return adjacent_cliques