from typing import Set, Tuple, Dict

from graph import GameGraph


class GameGraphSimple:
    REACHABILITY_PLAYER = "Ron"
    SAFETY_PLAYER = "Simon"

    def __init__(self, reach_nodes: Set[int], safety_nodes: Set[int], edges: Set[Tuple[int, int]]):
        self.straight: Dict[int, Set[int]] = dict()
        self.transpose: Dict[int, Set[int]] = dict()
        self.controlled_nodes: Dict[str, Set[int]] = dict()

        for n in reach_nodes.union(safety_nodes):
            self.straight[n] = set()
            self.transpose[n] = set()

        self.reachability_player_nodes = reach_nodes
        self.safety_player_nodes = safety_nodes

        for u, v in edges:  # (u -> v)
            self.add_edge(u, v)

    def add_edge(self, u: int, v: int):
        # add (u -> v) edge to straight graph
        self.straight[u].add(v)
        # Add (v -> u) edge to transpose graph
        self.transpose[v].add(u)

    def get_outbound_nodes(self, u: int) -> Set[int]:
        """
        Get all nodes reachable from u with an outbound edge
        :param u: a node in the graph
        :return: the set of reachable nodes
        """
        return self.straight[u]

    def get_inbound_nodes(self, u: int) -> Set[int]:
        """
        Get all nodes that can reach the given node and that are controlled by the given player
        :param u: a node in the graph
        :return: the set of nodes controlled by the given player that can have an outbound edge to the given node
        """
        return self.transpose[u]

    @property
    def nodes(self) -> Set[int]:
        return self.safety_player_nodes.union(self.reachability_player_nodes)

    def controller(self, u):
        return GameGraph.SAFETY_PLAYER if u in self.safety_player_nodes else GameGraph.REACHABILITY_PLAYER

    @staticmethod
    def from_regular_graph(G: GameGraph):
        edges = set()
        for n, n_edges in G.straight.items():
            for m in n_edges:
                edges.add((n, m))
        reach_nodes = G.reachability_player_nodes
        safe_nodes = G.safety_player_nodes
        return GameGraphSimple(reach_nodes, safe_nodes, edges)
