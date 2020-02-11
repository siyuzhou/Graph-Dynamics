import numpy as np
from graph import Graph


def interaction_func1(x, y):
    """Pair-wise repulsive interaction."""
    d = np.linalg.norm(y - x)
    return (x - y) / d / d / d  # Repulsion


def interaction_func2(x, y):
    """Cohesion squared."""
    d = np.linalg.norm(y - x)
    return (y - x) * d


def node_update_func(x, y):
    return x + y * 0.01


def random_connection_matrix(n, m):
    """
    Return a n x n binary matrix with m 1s in each row.
    """
    edges = np.zeros((n, n))
    particle_idxs = np.arange(n)
    for i in range(n):
        if m == 'y':
            k = np.random.randint(1, n)
        else:
            k = int(m)

        for j in np.random.choice(particle_idxs[particle_idxs != i], k, replace=False):
            edges[j, i] = 1  # j is i's target, thus j influences i through edge j->i.

    return edges


def random_init_states(n, k, scale=1.):
    """
    Return n x k state vectors.
    """
    return np.random.rand(n, k) * scale


def generate_timeseries(g, steps):
    states = [g.states]

    for _ in range(steps):
        g.update()
        states.append(g.states)

    return np.array(states)


def main():
    N = 10
    M = 3
    K = 2
    STEPS = 50
    init_states = random_init_states(N, K)
    connection_matrix = random_connection_matrix(N, M)

    g = Graph(connection_matrix, init_states)
    g.set_node_interaction(interaction_func1)
    g.set_neighbor_interaction(interaction_func2)
    g.set_node_update(node_update_func)

    timeseries = generate_timeseries(g, STEPS)

    np.save('data.npy', timeseries)


if __name__ == '__main__':
    main()
