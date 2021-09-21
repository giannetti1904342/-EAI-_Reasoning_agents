import random
from typing import Set, Tuple, List, Dict
import time
import math

from graph import GameGraph
from draw_graph import draw_graph


# Implementation of the "Multiple-perspective algorithm". The function is called "safety_game" because the
# algorithm starts solving the game as a safety game, but then it determines at each step if it is convenient
# to switch point of view according to an heuristic. Please refer the docs "The employed heuristic".
def safety_game(G: GameGraph, target_safe: Set[int], target_reach: Set[int], threshold=float('Inf'),
                draw=False, image_name="smart_{}.png", positions=None) -> Tuple[Set[int], List[str]]:
    
    #Initialize the winning set to target set for the safety player
    win: Set[int] = target_safe.copy()
    #Initialize the losing set as the complement of the winning set (this is actually the winning set for the reachability player)
    lose: Set[int] = {u for u in G.nodes if u not in win}

    #Inizialize a list which will contain the history of the perfomed step
    steps = []
    iteration = 0
    #Inizialize the last computed force reach as a copy of the losing set.     
    #Please refer to the subsection "Current set optimization" in the section "Optimizations" of the docs.
    last_force_reach: Set[int] = lose.copy()
    #Launch the routine to draw the graph
    if draw:
        draw_graph(G, target=target_reach, image_name=image_name.format(0), positions=positions)
    while True:
        iteration += 1
        #Update the cardinality of the winning set
        size_win = len(win)
        #Set up the condition for performing the step forward
        must_do_step_forward = size_win <= threshold

        #Step will contain the sequence of steps performed by the algorithm, either 'forward' or 'backward'
        step = 'forward' if must_do_step_forward else 'backward'
        #Print(f"Iteration {iteration}: step {step}")
        steps.append(step)

        #Case 1: [size_win <= threshold] -> Convenient to perform step forward
        if must_do_step_forward:
            #Compute the Force set through the step_forward
            force_set = step_forward(G, win)
            #Compute the new winning set as the intersection of the winning set and the Force set
            win_new = win.intersection(force_set)
            #Compute the losing set in the smartest way (by performing the symmetric difference operation)
            #(it computes all those elements which belongs either to the losing set or to the Force set but not to both)
            #i.e., Lose = Lose U (win.symmetric_difference(force_set))
            #In doing so, the update will be very fast.
            lose.update(win.symmetric_difference(force_set))
            #Update the winning set
            win = win_new
        else: # Case 2: [size_win > threshold] -> Convenient to perform step backward
            #Compute the Force set through the step_backward
            force_set = step_backward(G, lose, last_force_reach)
            #Compute the new losing set as the union of the losing set (i.e. the reachability player's winning set) and the Force set.
            lose.update(force_set)
            #Compute the new winning set for the safety player by removing all the nodes that are rechable for the reachability player (hence not safe). 
            win.difference_update(force_set)
            #Update the last_force_reach set (i.e., this is the Current set or in other terms the last computed force reach).
            last_force_reach = force_set

        #Launch the routine to draw the graph    
        if draw:
            draw_graph(G, target=target_reach, win=win, positions=positions,
                       image_name=image_name.format(iteration))
        #For the sake of efficiency, we compare the sets' cardinalities instead of comparing their elements to verify if the fixpoint has been reached.    
        if len(win) == size_win:
            return win, steps

#Function that implements the fundamental step to solve the game from the safety player's point of view.
#The underlying logic is the following:
#I'm in a safe node and I need to ensure to identify the set of nodes that allow me to remain in the winning set forever.
def step_forward(G: GameGraph, win: Set[int]) -> Set[int]:
    #Compute the Force set as the union of Reach_Comp(X) and Safety_Comp(X)
    f_safe: Set[int] = force_safe_forward(G, win)
    f_reach: Set[int] = force_reach_forward(G, win)
    return f_safe.union(f_reach)


def force_safe_forward(G: GameGraph, win) -> Set[int]:
    #Compute all the nodes present in the winning set controlled by the safety player
    win_safety = win.intersection(G.safety_player_nodes)
    #Initialize the safety compoment of the force set as the empty set
    f: Set[int] = set()
    #For every node in the previously computed set
    for u in win_safety:
        #Obtain its neighbors i.e. the nodes reachable from u with an outgoing edge
        neighbors = G.get_outbound_nodes(u)
        #In case the node is altredy safe and isolated, the node is safe
        if len(neighbors) == 0:
            #Then we add it to the force set.
            f.add(u)
        else:
            #For every node in the neighbors set
            for v in neighbors:
                #If the neighbor node is in the winning set, it is safe. Hence, we add it to the force set.
                if v in win:
                    f.add(u)
                    break
    return f


def force_reach_forward(G: GameGraph, win) -> Set[int]:
    #Compute the set of nodes in the winning set that are controlled by the reachability player
    win_reach = win.intersection(G.reachability_player_nodes)
    #Initialize the safety compoment of the force set as the empty set
    f: Set[int] = set()
    #For every node in the previously computed set
    for u in win_reach:
        #If the neighbors of the node u are contained in the winning set (i.e., the node can reach only safe nodes) the node is definitely safe. 
        #Hence, we add it to the force. 
        if G.get_outbound_nodes(u).issubset(win):
            f.add(u)
    return f

#Function that implements the fundamental step to solve the game from the reachability player's point of view.
#I.e. Its goal is to compute the set of nodes from which it is possible to reach the reachability player's target set.  
#Note since we have initially set up the game as a safety game, we refer to the reachability player's target set as the "Lose" set.
def step_backward(G: GameGraph, lose, last_force_reach) -> Set[int]:
    f_reach: Set[int] = force_reach_backward(G, lose, last_force_reach)
    f_safe: Set[int] = force_safe_backward(G, lose)
    return f_reach.union(f_safe)


def force_reach_backward(G: GameGraph, lose: Set[int], last_force_reach: Set[int]) -> Set[int]:
    #Initialize the force set to the empty set
    f: Set[int] = set()
    #For each node in the last computed force reach (i.e. the Current set)
    for u in last_force_reach:
        #Compute the nodes that:
        # - are controlled by the reachability player
        # - are rachable from the current set the transpose graph 
        # - are not already contained in the reachability player's winning set
        inbound = G.get_inbound_nodes(u, GameGraph.REACHABILITY_PLAYER).difference(lose)  #Huge time saving if remove Q! (~2x) 
        # The optimizations makes the update much much faster than F = F.union(inbound) (~6x)! 
        f.update(inbound)   
    return f


def force_safe_backward(G: GameGraph, lose: Set[int]) -> Set[int]:
    #Initialize the force set to the empty set
    f: Set[int] = set()
    #Initialize the processed set to the empty set
    processed: Set[int] = set()
    #For every node in the reachability player's winning set 
    for u in lose:
        #Compute all the nodes that:
        # - are controlled by the safety player
        # - are reachable from the reachability player's winning set in the tranpose graph
        # - can reach only nodes that are contained in the safety player's winning set
        for v in G.get_inbound_nodes(u, GameGraph.SAFETY_PLAYER):
            #The following operations aim to check if all the outgoing edges of the node in the straight graph 
            #end in nodes that belong to the target set of the safety player.
            #We do not process two times the same node thanks to the "Processed list optimization". 
            #Please refer to the homonymous subsection in the docs for further details.
            if v not in processed and v not in lose:
                #Store the nodes that the node v can reach in the straigth graph
                outbound = G.get_outbound_nodes(v)
                #If all these outgoing edges are contained in the winning set of the safety player, add it to the force set.
                if outbound.issubset(lose):
                    f.add(v)
                processed.add(v)
    return f

#Function that builds the graph of the example provided in the course slides.
def get_graph_from_slides():
    E = {(0, 1), (0, 3), (1, 0), (1, 2), (2, 1), (2, 5), (3, 4), (3, 6), (4, 0), (4, 7), (4, 8), (5, 7), (6, 7),
         (7, 6), (7, 8), (8, 5)}  # , (1, 3)} add to make all nodes reachable
    s = {0, 2, 4, 5, 6}
    r = {1, 3, 7, 8}
    return GameGraph(r, s, E), {0, 1, 2, 3, 6, 7, 8}

#Function that launch the multiple algorithms developed to solve the example provided in the course slides.
def example_from_slides():

    example_graph, target_safe = get_graph_from_slides()
    target_reach = [n for n in example_graph.nodes if n not in target_safe]
    threshold = math.ceil(len(example_graph.nodes)/2)

    print("==========\n Computing safety set for the example graph from the slides")

    print("\n\n\n----- Smart Algorithm:")
    example_smart_t0 = time.perf_counter()
    safe_smart_example = safety_game(example_graph, target_safe, target_reach, threshold=threshold)
    example_smart_t1 = time.perf_counter()
    print(
        f"\tThe smart algorithm found {len(safe_smart_example)} safe nodes in {example_smart_t1 - example_smart_t0:0.4f} seconds.")
    print(f"The safe nodes are: [{safe_smart_example}]")

    print("======= End of safety computation for example graph from the slides =======\n\n")

#Function that benchmark the algorithms developed to solve multiple instances of the safety problem by using a set randomly graphs as arenas.
def random_graph_safety(no_isolated=False):
    print("\n\n\n======= Begin safety computation for random graph ===")

    import random
    # seed=103, N=1000000, E=5000000 and T=50000 are a good example. 39 s to generate, 52 s to solve (40 iterations)
    # seed=103, N=10000, E=50000 and T=500 are a good example
    # seed=103, N=50, E=100 and T=5 are a good example (instantaneous)
    random.seed(103)
    N = 100000
    E = 150000
    T = 5000
    target_reach = set()
    while len(target_reach) < T:
        target_reach.add(random.randint(0, N - 1))
    target_safe = {n for n in range(N) if n not in target_reach}
    if T < 100: print(f"Target: {target_reach}\n")

    draw = N < 50


    graph_gen_ti = time.perf_counter()
    print(f"\tGenerating random graph with {N} nodes, {E} edges and {T} target nodes.")

    random_graph = GameGraph.generate_random_graph(N, E, no_isolated=no_isolated)
    # random_graph, target_safe = get_graph_from_slides()
    # target_reach = [n for n in random_graph.nodes if n not in target_safe]
    if draw:
        positions = draw_graph(random_graph, target=target_reach, image_name="graph.png")

    threshold_forward = math.ceil(len(random_graph.nodes))


    graph_gen_tf = time.perf_counter()
    print(f"\tRandom graph generated in {graph_gen_tf - graph_gen_ti:0.4f} seconds.")
    print(f"\t\tPlayer 'Reachability' controls {len(random_graph.reachability_player_nodes)}")
    print(f"\t\tPlayer 'Safety' controls {len(random_graph.safety_player_nodes)}\n\n")


    print("\n\n\n----- Forward Algorithm:")
    forward_t0 = time.perf_counter()

    if draw:
        safe_forward = safety_game(random_graph, target_safe, target_reach, threshold=threshold_forward,
                               draw=draw, image_name="forward_images/step_{}.png", positions=positions)
    else:
        safe_forward = safety_game(random_graph, target_safe, target_reach, threshold=threshold_forward)

    forward_t1 = time.perf_counter()
    print(
        f"\tThe forward algorithm found {len(safe_forward)} safe nodes in {forward_t1 - forward_t0:0.4f} seconds.")
    if len(safe_forward) <= 100: print(f"The safe nodes are: [{safe_forward}]")


    print("======= End of safety computation for Forward =======\n\n")

    threshold_backward = 1   # math.ceil(len(random_graph.nodes) / 2)

    print("\n\n\n----- Backward Algorithm:")
    backward_t0 = time.perf_counter()

    if draw:
        safe_backward = safety_game(random_graph, target_safe, target_reach, threshold=threshold_backward, draw=True,
                                    image_name="backward_images/step_{}.png", positions=positions)
    else:
        safe_backward = safety_game(random_graph, target_safe, target_reach, threshold=threshold_backward)

    backward_t1 = time.perf_counter()
    print(
        f"\tThe backward algorithm found {len(safe_backward)} safe nodes in {backward_t1 - backward_t0:0.4f} seconds.")
    if len(safe_backward) <= 100: print(f"The safe nodes are: [{safe_backward}]")




    print("\n\n\n----- Optimized Algorithm:")
    t0 = time.perf_counter()
    threshold = math.ceil(len(random_graph.nodes) / 2)
    if draw:
        safe = safety_game(random_graph, target_safe, target_reach, threshold=threshold,
                                   draw=draw, image_name="forward_images/step_{}.png", positions=positions)
    else:
        safe = safety_game(random_graph, target_safe, target_reach, threshold=threshold)

    t1 = time.perf_counter()
    print(
        f"\tThe optimized algorithm found {len(safe)} safe nodes in {t1 - t0:0.4f} seconds.")
    if len(safe) <= 100: print(f"The safe nodes are: [{safe}]")

    print("======= End of safety computation for Optimized =======\n\n")






    print("======= End of safety computation for random graph =======\n\n")


#Function that generates a random graph given: 
#- Number of nodes
#- Number of edges
#- Size of the target set
def generate_random_graph(num_nodes: int, num_edges: int, target_safe_size:int) -> Tuple[GameGraph, Set[int], Set[int]]:
    target_safe = set()
    while len(target_safe) < target_safe_size:
        target_safe.add(random.randint(0, num_nodes - 1))
    target_reach = {n for n in range(num_nodes) if n not in target_safe}

    graph_gen_ti = time.perf_counter()
    print(f"Generating random graph with {num_nodes} nodes, {num_edges} edges and {target_safe_size} target-safe nodes.")
    random_graph = GameGraph.generate_random_graph(num_nodes, num_edges, no_isolated=True)\

    graph_gen_tf = time.perf_counter()
    print(f"\tRandom graph generated in {graph_gen_tf - graph_gen_ti:0.4f} seconds.")
    print(f"\t\tPlayer 'Reachability' controls {len(random_graph.reachability_player_nodes)}")
    print(f"\t\tPlayer 'Safety' controls {len(random_graph.safety_player_nodes)}\n\n")

    return random_graph, target_safe, target_reach

#Utility function that generates a random float
def randfloat(a: float, b: float):
    return a + (b-a)*random.random()

#Function that launches multiple experiments and provide the associated statistics.
def test(num_tests: int, num_nodes_min: int, num_nodes_max,
         avg_edges_per_node_min: float, avg_edges_per_node_max: float,
         target_safe_ratio_min: float, target_safe_ratio_max:float, random_seed=103):
    random.seed = random_seed

    statistics = []
    for i in range(1, num_tests+1):
        print(f"\n\n===================\n"
                   f"== EXPERIMENT {i} ==\n"
                    "===================\n\n")
        num_nodes: int = random.randint(num_nodes_min, num_nodes_max)
        num_edges: int = math.ceil(num_nodes * randfloat(avg_edges_per_node_min, avg_edges_per_node_max))
        target_safe_size: int = math.floor(num_nodes * randfloat(target_safe_ratio_min, target_safe_ratio_max))

        statistics += [random_graph_safety_experiment(num_nodes, num_edges, target_safe_size)]

    #is_always_faster = all([s['is_optimized_faster'] for s in statistics])

    avg_time_saving_wrt_forward = 1/num_tests * sum({s['time_saving_wrt_forward'] for s in statistics}) * 100.0
    avg_time_saving_wrt_backward = 1/num_tests * sum({s['time_saving_wrt_backward'] for s in statistics}) * 100.0
    avg_time_saving_wrt_best = 1/num_tests * sum({s['time_saving_wrt_best'] for s in statistics}) * 100.0

    print(f"\n\n\n ==== FINAL RESULTS ===\n"
          #f" - The optimized algorithm is {'' if is_always_faster else 'not'} always faster than the naive versions!\n\n"
          f" - The average time saving wrt the forward algorithm is {avg_time_saving_wrt_forward:0.2f}%\n"
          f" - The average time saving wrt the backward algorithm is {avg_time_saving_wrt_backward:0.2f}%\n"
          f" - The average time saving wrt the best naive algorithm is {avg_time_saving_wrt_best:0.2f}%")


#Function empoyed in the function "test" to compute the statistics.
def random_graph_safety_experiment(num_nodes: int, num_edges: int, target_safe_size:int):
    random_graph: GameGraph
    target_safe: Set[int]
    target_reach: Set[int]
    random_graph, target_safe, target_reach = generate_random_graph(num_nodes, num_edges, target_safe_size)

    forward_results = safety_run(random_graph, target_safe, target_reach, len(random_graph.nodes), 'Forward')
    backward_results = safety_run(random_graph, target_safe, target_reach, 1, 'Backward')
    optimized_results = safety_run(random_graph, target_safe, target_reach, math.ceil(len(random_graph.nodes) / 2),
                                   'Optimized')

    statistics: Dict[str, float] = dict()
    forward_time = forward_results[3]
    backward_time = backward_results[3]
    optimized_time = optimized_results[3]
    statistics['forward_time'] = forward_time
    statistics['backward_time'] = backward_time
    statistics['optimized_time'] = optimized_time

    tolerance: float = 0.1
    is_optimized_faster: bool = abs(optimized_time-forward_time) <= tolerance and abs(optimized_time-backward_time) <= tolerance

    statistics['is_optimized_faster'] = is_optimized_faster

    statistics['time_saving_wrt_forward'] = (forward_time - optimized_time)/forward_time
    statistics['time_saving_wrt_backward'] = (backward_time - optimized_time)/backward_time
    best_naive_time: float = min(forward_time, backward_time)
    statistics['time_saving_wrt_best'] = (best_naive_time - optimized_time)/best_naive_time

    statistics['forward_steps_in_optimized'] = optimized_results[1]
    statistics['backward_steps_in_optimized'] = optimized_results[2]

    return statistics



#Utility function employed in the function "random_graph_safety_experiment"
def safety_run(random_graph: GameGraph, target_safe: Set[int], target_reach: Set[int], threshold: int, run_name: str):
    print(f"\t\n\n======= {run_name} Algorithm: =======")
    t0 = time.perf_counter()

    safe, steps = safety_game(random_graph, target_safe, target_reach, threshold=threshold)

    t1 = time.perf_counter()
    total_time = t1 - t0
    print(
        f"\t\tThe {run_name} algorithm found {len(safe)} safe nodes in {len(steps)} iterations over {total_time:0.4f} "
        f"seconds.")

    print(f"\t======= End of safety computation for {run_name} =======\n\n")

    forward_steps: int = len({s for s in steps if s == 'forward'})
    backward_steps: int = len(steps) - forward_steps

    return len(steps), forward_steps, backward_steps, total_time



#Main
if __name__ == '__main__':
    # example_from_slides()
    # random_graph_safety(no_isolated=True)
    num_tests = 1
    N_min = 1e5
    N_max = 1e6
    test(num_tests, N_min, N_max, 1, 5, 0.1, 0.5, 103)
