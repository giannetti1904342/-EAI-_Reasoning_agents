import networkx as nx
import matplotlib.pyplot as plt
from graph import GameGraph

#Routine to draw graphs
def draw_graph(graph, positions=None, target=None, win=None, image_name='graph.png'):
    #Initialize the target and the winning set 
    if target is None:
        target = []
    if win is None:
        win = []

    #Generate a graph using NetworkX
    G = nx.DiGraph()
    
    #For every node in the set of nodes specified, add the node to the graph
    for u in graph.nodes:
        G.add_node(u)

    #Initialize the set of edges
    edges = []

    #For every node in the set of nodes specified, add the node to the graph
    for n in graph.nodes:
        #Generate the set of edges of the node n by using the function that return all the nodes reachable from u with an outbound edge.
        #(shortcut to generate the set of the edges of the node n since the nodes are labelled with an integer)
        edges_n = graph.get_outbound_nodes(n)
        #For all the edges in the previously generated set
        for m in edges_n:
            #Add the edge to the graph
            G.add_edge(n, m)
            #Include the edge into the set of edges as a tuple
            edges.append((n, m))
    #This is required by Network X to specify the colors of the nodes        
    col_map = {
                GameGraph.REACHABILITY_PLAYER: 'red',
               GameGraph.SAFETY_PLAYER: 'green',
               'target': 'blue',
               'win': 'purple'}

    #This is required by Network X to specify the shapes of the nodes
    shape_map = { GameGraph.REACHABILITY_PLAYER: 'r',
               GameGraph.SAFETY_PLAYER: 's'}

    #Initialize the required lists           
    colors = []
    shapes = []
    #For all the nodes in the graph, append to the previously generated lists the shapes and the colors according to the set to which the nodes belongs
    for n in G.nodes():
        shapes.append(shape_map[graph.controller(n)])
        if n in target:
            colors.append(col_map['target'])
        elif n in win:
            colors.append(col_map['win'])
        else:
            colors.append(col_map[graph.controller(n)])

    #If the positions are not generated, generate an adequate set of positions        
    if not positions:
        positions = nx.spring_layout(G, k=0.75, iterations=20)

    #Draw the nodes, their labels and the edges    
    nx.draw_networkx_nodes(G, positions, cmap=plt.get_cmap('jet'),
                           node_color=colors, node_size=500)#, node_shape=shapes)
    nx.draw_networkx_labels(G, positions)
    nx.draw_networkx_edges(G, positions, edgelist=edges, arrows=True)
    #Show the figure, use this option for python notebooks
    # plt.show()
    #Save the figure
    plt.savefig(image_name)
    #Clear the entire current figure with its axes
    plt.clf()
    #Return the generated positions
    return positions
