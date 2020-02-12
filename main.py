import functools
import argparse
import numpy as np
from graph import Graph


def node_interaction_func(x, y, scale=1):
    """Pair-wise repulsive interaction."""
    # x, y has shape (N, 4)
    xr = np.take(x, [0, 1], -1)
    yr = np.take(y, [0, 1], -1)
    d = np.linalg.norm(yr - xr)
    return scale * (xr - yr) / d / d  # Repulsion


def neighbor_interaction_func(x, y, scale=1):
    """Cohesion squared."""
    # x, y has shape (N, 4)
    xr = np.take(x, [0, 1], -1)
    yr = np.take(y, [0, 1], -1)
    d = np.linalg.norm(yr - xr)
    return scale * (yr - xr) * np.sqrt(d)


def null_interaction(x, y):
    return np.zeros(x.shape[:-1] + (2,))


def node_update_func(x, y, dt=1):
    # Last dimension of x: 4, y: 2
    xr = np.take(x, [0, 1], -1)
    xv = np.take(x, [2, 3], -1)

    xr += xv * dt + 0.5 * y * dt * dt
    xv += y * dt

    return np.concatenate([xr, xv], -1)


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


def full_connection_matrix(n):
    cm = np.ones((n, n))
    np.fill_diagonal(cm, 0)
    return cm


def random_init_states(n, k, scale=1.):
    """
    Return n x k state vectors.
    """
    return (np.random.rand(n, k) - 0.5) * scale


def generate_timeseries(g, steps):
    states = [g.states]

    for _ in range(steps):
        g.update()
        states.append(g.states)

    return np.array(states)


def main():
    SCALE = 10
    init_states = random_init_states(ARGS.n, ARGS.k, SCALE)
    connection_matrix = random_connection_matrix(ARGS.n, ARGS.m)
    # connection_matrix = full_connection_matrix(N)

    g = Graph(connection_matrix, init_states)
    g.set_node_interaction(functools.partial(node_interaction_func, scale=1))
    # g.set_node_interaction(null_interaction)
    g.set_neighbor_interaction(functools.partial(neighbor_interaction_func, scale=0.3))
    # g.set_neighbor_interaction(null_interaction)
    g.set_node_update(functools.partial(node_update_func, dt=0.1))

    data = []
    for i in range(ARGS.instances):
        timeseries = generate_timeseries(g, ARGS.steps)
        if (i+1) % 100 == 0:
            print(f'{i+1}/{ARGS.instances} done.')
        data.append(timeseries)
    print('All done.')

    data = np.asarray(data)
    np.save('data.npy', data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', type=int, default=10,
                        help='number of particles')
    parser.add_argument('-m', type=int, default=5,
                        help='number of neighbors for each particle')
    parser.add_argument('-k', type=int, default=4,
                        help='size of state vector')
    parser.add_argument('-s', '--steps', type=int, default=50,
                        help='number of timesteps of a trajectory')
    parser.add_argument('-i', '--instances', type=int, default=1,
                        help='number of simulation instances')

    ARGS = parser.parse_args()

    main()
