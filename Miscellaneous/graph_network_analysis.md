# Graph Network Analysis

## Number of edges

Calculating the distribution of the in-degree and out-degree in directed graphs (or distribution of degree in un-directed graphs) can be useful for understanding connections of node.

## Average Path Length

The average of the shortest path lengths for all possible node pairs. Gives a measure of ‘tightness’ of the Graph and can be used to understand how quickly/easily something flows in this Network.

## BFS and DFS

Breadth first search and Depth first search are two different algorithms used to search for Nodes in a Graph. They are typically used to figure out if we can reach a Node from a given Node. This is also known as Graph Traversal.

## Centrality

Centrality aims to find the most important nodes in a network. There may be different notions of “important” and hence there are many centrality measures. 

Some of the most commonly used ones are:

    1. Degree Centrality – This is the number of edges connected to a node. In the case of a directed graph, we can have 2 degree centrality measures. Inflow and Outflow Centrality.
    2. Closeness Centrality – Of a node is the average length of the shortest path from the node to all other nodes
    3. Betweenness Centrality – Number of times a node is present in the shortest path between 2 other nodes

These centrality measures have variants and the definitions can be implemented using various algorithms. All in all, this means a large number of definitions and algorithms.

## Network Density

A measure of the structure of a network, which measure how many links from all possible links within the network are realized. The density is 0 if there are no edges and 1 for a complete Graph.
 
## Scale-Free Property

'Real' networks have a certain underlying creation process, which results in some nodes with a much higher degree compared to other nodes.

These nodes with a very high degree in the network are called hubs. Example can be of Twitter as a Social Network where prominent people represent hubs, having much more edges to other nodes than the average user.

## Node Connectivity

This describes the number of nodes one must delete from the Graph until it is disconnected. Connected means that if every node in a graph can reach any other node in the network via edges. If this is not the case the graph is disconnected. An important property of any graph should to be that it is not easily to disconnect. 
