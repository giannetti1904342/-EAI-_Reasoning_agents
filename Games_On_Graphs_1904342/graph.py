from typing import Set, Tuple, Dict
import random


class GameGraph:
    REACHABILITY_PLAYER = "Ron"
    SAFETY_PLAYER = "Simon"

    def __init__(self, reach_nodes: Set[int], safety_nodes: Set[int], edges: Set[Tuple[int, int]]):
        """
        Every node must be controlled by either Reachability or Safety player
        Example of data structures:
          straight = {1: {2,3,4},  2: {4}, 3: {2}, 4: {}}
          transpose = {1: {}, 2: {"reach": {1},"safe": {3}}, 3: {"reach": {1},"safe":{}}, 4: {"reach": {1}, "safe":{}}}
        """

        self.nodes: Set[int] = reach_nodes.union(safety_nodes)
        self.straight: Dict[int, Set[int]] = dict()
        self.transpose: Dict[int, Dict[str, Set[int]]] = dict()

        self.reachability_player_nodes: Set[int] = reach_nodes
        self.safety_player_nodes: Set[int] = safety_nodes

        for n in reach_nodes.union(safety_nodes):
            self.straight[n] = set()
            self.transpose[n] = {GameGraph.REACHABILITY_PLAYER: set(), GameGraph.SAFETY_PLAYER: set()}
        for u, v in edges:
            self.add_edge(u, v)

    def add_edge(self, u: int, v: int) -> bool:
        """
        Adds a new edge to the graph, but only if the two nodes are already in it
        :param u: a node
        :param v: a node, different from u
        :return: True if the add was successful, False otherwise
        """
        if u not in self or v not in self or u == v or v in self.straight[u]:
            return False
        # add (u -> v) edge to straight graph
        self.straight[u].add(v)

        # Add (v -> u) edge to transpose graph, according to u's controller
        if u in self.reachability_player_nodes:
            self.transpose[v][GameGraph.REACHABILITY_PLAYER].add(u)
        elif u in self.safety_player_nodes:
            self.transpose[v][GameGraph.SAFETY_PLAYER].add(u)
        else:
            raise Exception(f"You have not defined node {u} as controlled by Safety or Reachability player.")
        return True

    def get_outbound_nodes(self, u: int) -> Set[int]:
        """
        Get all nodes reachable from u with an outbound edge
        :param u: a node in the graph
        :return: the set of reachable nodes
        """
        return self.straight[u]

    def get_inbound_nodes(self, u: int, player: str) -> Set[int]:
        """
        Get all nodes that can reach the given node and that are controlled by the given player
        :param u: a node in the graph
        :param player: a string representing one of the two players, "reach" or "safe"
        :return: the set of nodes controlled by the given player that can have an outbound edge to the given node
        """
        return self.transpose[u][player]

    def __contains__(self, u):
        return u in self.reachability_player_nodes or u in self.safety_player_nodes

    def controller(self, u):
        return GameGraph.SAFETY_PLAYER if u in self.safety_player_nodes else GameGraph.REACHABILITY_PLAYER

    @staticmethod
    def generate_random_graph(N: int, E: int, self_edges=True, no_isolated=False):
        nodes = {n for n in range(N)}  # 0..N-1
        k = random.randint(1, N - 1)  # at least one node for each player
        reach_nodes = {n for n in nodes if n < k}  # {n | n \in N and n <= k}
        safe_nodes = nodes.difference(reach_nodes)
        edges = set()
        while len(edges) < E:
            u = random.randint(0, N - 1)
            v = random.randint(0, N - 1)
            if u == v and not self_edges:
                continue
            if (u, v) not in edges:
                edges.add((u, v))

        graph = GameGraph(reach_nodes, safe_nodes, edges)

        if no_isolated:
            for n in graph.nodes:
                if not graph.get_inbound_nodes(n, GameGraph.REACHABILITY_PLAYER) and \
                        not graph.get_inbound_nodes(n, GameGraph.SAFETY_PLAYER) and \
                        not graph.get_outbound_nodes(n):

                    if len(graph.straight[n]) == 0:
                        v = random.randint(0, N - 1)
                        # keep sampling v until it is different from n
                        while v == n:
                            v = random.randint(0, N - 1)

                        # choose edge direction randomly
                        coin = random.randint(0, 1)

                        graph.add_edge(n, v) if coin else graph.add_edge(v, n)
        return graph

    @staticmethod
    def get_graph_from_slides():
        E = {(0, 1), (0, 3), (1, 0), (1, 2), (2, 1), (2, 5), (3, 4), (3, 6), (4, 0), (4, 7), (4, 8), (5, 7), (6, 7),
             (7, 6), (7, 8), (8, 5)}  # , (1, 3)} add to make all nodes reachable
        s = {0, 2, 4, 5, 6}
        r = {1, 3, 7, 8}
        return GameGraph(r, s, E), {0, 1, 2, 3, 6, 7, 8}


if __name__ == '__main__':
    example_graph, _ = GameGraph.get_graph_from_slides()
    print(example_graph.straight)
    print(example_graph.transpose)
