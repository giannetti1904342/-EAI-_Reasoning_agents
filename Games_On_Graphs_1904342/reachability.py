from typing import Set
import time

from graph import GameGraph
from graph_simple import GameGraphSimple
from draw_graph import draw_graph

#Improved backward algorithm implementation. Please refer the documentation for further details.
def reachability_game(G: GameGraph, target: Set[int], draw=False, image_name="smart_{}.png") -> Set[int]:
    """
    Given a graph and a set of target nodes, returns the set of nodes such that
    the target is not reachable from them. Optionally, writes the steps of the algorithm as pictures.
    :param G: a graph
    :param target: a set of integers
    :param draw: a boolean
    :param image_name: the name of the file where to write the images. Must be formattable with the iteration number
    :return: a set of integers representing safe nodes
    """
    #Compute the positions of the nodes to use the same positions of the nodes for all the images (for the sake of clarity).
    if draw:
        positions = draw_graph(G, target=target, image_name=image_name.format(0))
    else:
        positions = None
    
    #Using Do-While construction
    #Initialize target set
    Q: Set[int] = target.copy()
    #Initialize the Current set (i.e. set of the last nodes found reachable). Please refer to the documentation for further details.
    C: Set[int] = target.copy()
    #Compute Reach_comp(X) (i.e. the component related to the reachability player)
    F_reach: Set[int] = force_reach(G, Q, C)
    #Compute Safety_comp(X) (i.e. the component related to the safety player)
    F_safe: Set[int] = force_safe(G, Q)
    #Compute the Force set
    F: Set[int] = F_reach.union(F_safe)
    #Compute the new target set
    Q_prime: Set[int] = Q.union(F)
    #Routine to draw the graph at the first iteration
    iteration = 1
    if draw:
        draw_graph(G, target=target, win=Q_prime, positions=positions, image_name=image_name.format(iteration))
    
    #While the fixpoint is not reached
    while Q != Q_prime:
        print(f"Iteration: {iteration} - Q: {len(Q)}")
        #Update the set of the last nodes found reachable to the last computed force
        C = F
        #Update the target set
        Q = Q_prime
        #Compute the Force set
        F_reach: Set[int] = force_reach(G, Q, C)
        F_safe: Set[int] = force_safe(G, Q)
        F: Set[int] = F_reach.union(F_safe)
        #Compute the new target set
        Q_prime = Q.union(F)
        #Routine to draw the graph
        iteration += 1
        if draw:
            draw_graph(G, positions=positions, target=target, win=Q_prime,
                       image_name=image_name.format(iteration))

    #Routine to draw the graph at the last iteration        
    if draw:
        draw_graph(G, positions=positions, win=Q_prime, image_name=image_name.format("final"))
    return Q

#Optimized function whose goal is to compute Reach_Comp(X). Precisely, it computes the set of nodes controlled by the reachability
#player that are reachable from C in the transpose graph. This is actually equivalent to compute all the nodes 
#from which the reachability player can reach C (), but faster. Please refer to the documentation for further details.
def force_reach(G: GameGraph, Q: Set[int], C: Set[int]) -> Set[int]:
    #Initialize the Force set to the empty set
    F: Set[int] = set()
    #Check all the nodes in the set of the last nodes found reachable.
    #Please refer to the paragraph "Current set optimization" in the section "Optimizations" for further details.
    for u in C:
        #Compute the set of nodes that:
        # - are controlled by the reachability player
        # - are reachable from the current set in the transpose graph
        # - are not already contained in the winning set of the reachability player
        #This last point has been added to avoid adding nodes already present in the winning set to the Force
        inbound = G.get_inbound_nodes(u, GameGraph.REACHABILITY_PLAYER).difference(Q) # -> Huge time saving by removing Q (~2x)
        #In fact, by combinining this optimization with the usage of F.update(inbound) instead of F.union(inbound)
        #We obtain that the update is much much faster (~6x)!
        F.update(inbound)
    return F

# Optimized function whose goal is to compute Safety_Comp(X), i.e., the set of nodes controlled by the safety player for which
# the safety player cannot avoid to enter in the reachability player's winning setthat.
def force_safe(G: GameGraph, Q: Set[int]) -> Set[int]:
    #Initialize the force_set to the empty set
    F: Set[int] = set()
    #Initialize the set of processed nodes
    processed: Set[int] = set()
    #Check all the nodes in the winning set
    for u in Q:
        #Compute the set of nodes that:
        # - are controlled by the safety player
        # - are reachable from the reachability player's winning set in the transpose graph
        for v in G.get_inbound_nodes(u, GameGraph.SAFETY_PLAYER):
            #If the node has not been processed and it is not in the target set
            if v not in processed and v not in Q:
                #Compute the set of nodes that the node v can reach in the straight graph (i.e. the neighbors of v in the straigh graph)
                outbound = G.get_outbound_nodes(v)
                #Check if the previously computed set is a subset of the target set
                #In case add the node to the Force set and to the processed set to avoid to check it again
                #Please refer to the paragraph "Processed list optimization" in the section "Optimizations"
                if outbound.issubset(Q):
                    F.add(v)
                processed.add(v)
    return F

# Un-optimized version of the backward algorithm.  
def reachability_game_naive(G: GameGraphSimple, target: Set[int], draw=False, image_name="naive_{}.png") -> Set[int]:
    """
    Naive version, directly derived from the slides.
    Given a graph and a set of target nodes, returns the set of nodes such that
    the target is not reachable from them. Optionally, writes the steps of the algorithm as pictures.
    :param G: a graph
    :param target: a set of integers
    :param draw: a boolean
    :param image_name: the name of the file where to write the images. Must be formattable with the iteration number
    :return: a set of integers representing safe nodes
    """
    positions = None
    if draw:
        positions = draw_graph(G, target=target, image_name=image_name.format(0))

    Q: Set[int] = target.copy()
    F_reach: Set[int] = force_reach_naive(G, Q)
    F_safe: Set[int] = force_safe_naive(G, Q)
    F: Set[int] = F_reach.union(F_safe)
    Q_prime: Set[int] = Q.union(F)

    iteration = 1
    if draw:
        draw_graph(G, target=target, win=Q_prime, positions=positions, image_name=image_name.format(iteration))
    while Q != Q_prime:
        Q = Q_prime
        F_reach: Set[int] = force_reach_naive(G, Q)
        F_safe: Set[int] = force_safe_naive(G, Q)
        F: Set[int] = F_reach.union(F_safe)
        Q_prime = Q.union(F)
        if draw:
            iteration += 1
            draw_graph(G, positions=positions, target=target, win=Q_prime,
                       image_name=image_name.format(iteration))

    if draw:
        iteration += 1
        draw_graph(G, positions=positions, target=target, win=Q_prime, image_name=image_name.format(iteration))
    return Q


# Un-optimized function whose goal is to compute Reach_Comp(X). We avoid commenting the code to provide the reader a lean code.
def force_reach_naive(G: GameGraphSimple, Q: Set[int]) -> Set[int]:
    F: Set[int] = set()
    for u in G.nodes:
        if G.controller(u) == GameGraph.REACHABILITY_PLAYER:
            reachable = G.get_outbound_nodes(u)
            for q in Q:
                if q in reachable:
                    F.add(u)
                    break
    return F


# Un-optimized function whose goal is to compute Safety_Comp(X). We avoid commenting the code to provide the reader a lean code.
def force_safe_naive(G: GameGraphSimple, Q: Set[int]) -> Set[int]:
    F: Set[int] = set()
    for u in G.nodes:
        if G.controller(u) == GameGraph.SAFETY_PLAYER:
            reachable = G.get_outbound_nodes(u)
            if len(reachable) > 0:
                if reachable.issubset(Q):
                    F.add(u)
    return F


#This function reproduces the example taken from the course's slides.
def example_from_slides(naive=True):
    draw = False
    E = {(0, 1), (0, 3), (1, 0), (1, 2), (2, 1), (2, 5), (3, 4), (3, 6), (4, 0), (4, 7), (4, 8), (5, 7), (6, 7),
         (7, 6), (7, 8), (8, 5)}  # , (1, 3)} add to make all nodes reachable
    s = {0, 2, 4, 5, 6}
    r = {1, 3, 7, 8}
    target = {4, 5}
    example_graph = GameGraph(r, s, E)

    print("==========\n Computing safety set for the example graph from the slides")

    if naive:
        example_graph_simple = GameGraphSimple.from_regular_graph(example_graph)

        print("\n----- Naive Algorithm:")
        example_naive_t0 = time.perf_counter()
        safe_naive_example = reachability_game_naive(example_graph_simple, target,
                                               draw=draw, image_name="./example_naive_images/step_{}.png")
        example_naive_t1 = time.perf_counter()
        print(
            f"\tThe Naive algorithm found {len(safe_naive_example)} safe nodes in {example_naive_t1 - example_naive_t0:0.4f} seconds.")
        print(f"\t The safe nodes are:\n [{safe_naive_example}]")

    print("\n\n\n----- Smart Algorithm:")
    example_smart_t0 = time.perf_counter()
    safe_smart_example = reachability_game(example_graph, target, draw=draw, image_name="./example_smart_images/step_{}.png")
    example_smart_t1 = time.perf_counter()
    print(
        f"\tThe smart algorithm found {len(safe_smart_example)} safe nodes in {example_smart_t1 - example_smart_t0:0.4f} seconds.")
    print(f"The safe nodes are: [{safe_smart_example}]")

    print("======= End of safety computation for example graph from the slides =======\n\n")

#This function launches a bettery of automatic experiments on randomly generated graphs to benchmark the implemented algorithms.
def random_graph_reachability(draw: bool, naive=True, no_isolated=False):
    print("\n\n\n======= Begin safety computation for random graph ===")

    import random
    # seed=103, N=1000000, E=5000000 and T=50000 are a good example. 39 s to generate, 52 s to solve (40 iterations)
    # seed=103, N=10000, E=50000 and T=500 are a good example
    # seed=103, N=50, E=100 and T=5 are a good example (instantaneous)
    random.seed(103)
    N = 10000
    E = 20000    
    T = 5000
    target = set()
    while len(target) < T:
        target.add(random.randint(0, N - 1))
    if T < 100: print(f"Target: {target}\n")


    graph_gen_ti = time.perf_counter()
    print(f"\tGenerating random graph with {N} nodes, {E} edges and {T} target nodes.")
    random_graph = GameGraph.generate_random_graph(N, E, no_isolated=no_isolated)

    graph_gen_tf = time.perf_counter()
    print(f"\tRandom graph generated in {graph_gen_tf - graph_gen_ti:0.4f} seconds.")
    print(f"\t\tPlayer 'Reachability' controls {len(random_graph.reachability_player_nodes)}")
    print(f"\t\tPlayer 'Safety' controls {len(random_graph.safety_player_nodes)}\n\n")


    if naive:
        print(f"Generating equivalent naive graph...")
        random_graph_naive = GameGraphSimple.from_regular_graph(random_graph)
        print("Done.\n")

        print("\n----- Naive Algorithm:")
        naive_t0 = time.perf_counter()
        safe_naive = reachability_game_naive(random_graph_naive, target, draw=draw,
                                       image_name="./random_naive_images/step_{}.png")

        naive_t1 = time.perf_counter()
        print(
            f"\tThe Naive algorithm found {len(safe_naive)} safe nodes in {naive_t1 - naive_t0:0.4f} seconds.")
        print(f"\t The safe nodes are:\n [{safe_naive}]")

    print("\n\n\n----- Smart Algorithm:")
    smart_t0 = time.perf_counter()
    safe_smart = reachability_game(random_graph, target, draw=draw, image_name="./random_smart_images/step_{}.png")
    smart_t1 = time.perf_counter()
    print(
        f"\tThe smart algorithm found {len(safe_smart)} safe nodes in {smart_t1 - smart_t0:0.4f} seconds.")
    if len(safe_smart) <= 100: print(f"The safe nodes are: [{safe_smart}]")

    print("======= End of safety computation for random graph =======\n\n")

    if naive and safe_smart != safe_naive:
        print(" !!!!!!!!!!!!!!!!!!!!!!!!\nThe smart and naive algorithm return different safe sets!")
        print(f"\tNodes safe for Smart but not for Naive: [{safe_smart.difference(safe_naive)}\n")
        print(f"\tNodes safe for Naive but not for Smart: [{safe_naive.difference(safe_smart)}\n")
        print("This must definetely be caused by a bug. Look for it!")

#Main
if __name__ == '__main__':
    # example_from_slides(naive=True)
    random_graph_reachability(False, naive=True, no_isolated=True)
