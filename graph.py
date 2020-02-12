import numpy as np
import utils


class Graph:
    def __init__(self, connection_matrix, init_states):
        n = len(connection_matrix)
        if np.shape(connection_matrix) != (n, n):
            raise ValueError("'connection_matrix' must b a square matrix")

        if not np.isin(connection_matrix, (0, 1)).all():
            raise ValueError("'connection_matrix' elements must be 0 or 1")

        self.connection_matrix = np.array(connection_matrix)
        edge_sources, edge_targets = np.where(self.connection_matrix)
        self.edge_sources = utils.one_hot(edge_sources, n)
        self.edge_targets = utils.one_hot(edge_targets, n)

        self.states = np.array(init_states)
        if len(self.states.shape) != 2 or self.states.shape[0] != n:
            raise ValueError(
                "'init_states' must be vector states of uniform length for all nodes")

        self._node_interaction = None
        self._neighbor_interaction = None
        self._node_update = None

    def set_node_interaction(self, func):
        self._node_interaction = func

    def set_neighbor_interaction(self, func):
        self._neighbor_interaction = func

    def set_node_update(self, func):
        self._node_update = func

    def update(self):
        sources = np.dot(self.edge_sources, self.states)
        targets = np.dot(self.edge_targets, self.states)

        # msg from node interaction along edges
        try:
            edge_msgs = self._node_interaction(targets, sources)
        except TypeError:
            raise NameError("node_interaction function not defined")
        aggr_edge_msgs = np.dot(self.edge_targets.T, edge_msgs)

        # msg from interaction with neighbor averages
        neighbor_sums = np.dot(self.edge_targets.T, sources)
        neighbor_counts = np.sum(self.edge_targets.T, axis=1, keepdims=True)
        neighbor_counts[neighbor_counts == 0] = 1
        neighbor_avgs = neighbor_sums / neighbor_counts

        try:
            neighbor_msgs = self._neighbor_interaction(self.states, neighbor_avgs)
        except TypeError:
            raise NameError("neighbor_interaction function not defined")

        # Combine all msgs.
        all_msgs = aggr_edge_msgs + neighbor_msgs

        try:
            self.states = self._node_update(self.states, all_msgs)
        except TypeError:
            raise NameError("node_update function not defined")
