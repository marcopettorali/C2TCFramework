import networkx as nx

def find_paths(graph, source, target):
    """
    Finds all simple paths between a source and target node in the graph.

    Args:
        graph (networkx.Graph): The topology graph.
        source (str): The source node.
        target (str): The target node.

    Returns:
        list: A list of paths, where each path is represented as a list of edges with source, destination, and link info.
    """
    all_shortest_paths = list(nx.all_simple_paths(graph, source, target))

    # Convert paths to edges with labels
    all_shortest_links = [[{"src": u, "dest": v, "info": graph[u][v]} for u, v in zip(path, path[1:])] for path in all_shortest_paths]

    return all_shortest_links