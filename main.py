import os
import functools
import time
import multiprocessing
import argparse
import numpy as np
from graph import Graph
from functions import node_interaction_func, neighbor_interaction_func, node_update_func


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


def simulation(_):
    np.random.seed()
    SCALE = 10
    init_states = random_init_states(ARGS.n, ARGS.k, SCALE)
    connection_matrix = random_connection_matrix(ARGS.n, ARGS.m)

    g = Graph(connection_matrix, init_states)
    g.set_node_interaction(functools.partial(node_interaction_func, scale=1))
    g.set_neighbor_interaction(functools.partial(neighbor_interaction_func, scale=0.3))

    g.set_node_update(functools.partial(node_update_func, dt=0.1))
    states = [g.states]

    for _ in range(ARGS.steps):
        g.update()
        states.append(g.states)

    return np.array(states), connection_matrix


def run_simulation(simulation, instances, processes=1, batch=100, silent=False):
    pool = multiprocessing.Pool(processes=processes)
    data_all = []
    cm_all = []

    remaining_instances = instances

    prev_time = time.time()
    while remaining_instances > 0:
        n = min(remaining_instances, batch)
        data_pool = pool.map(simulation, range(n))

        data_pool, cm_pool = zip(*data_pool)

        remaining_instances -= n
        if not silent:
            print('Simulation {}/{}... {:.1f}s'.format(instances - remaining_instances,
                                                       instances, time.time()-prev_time))
        prev_time = time.time()

        data_all.extend(data_pool)
        cm_all.extend(cm_pool)

    return data_all, cm_all


def main():
    if ARGS.m != 'y' and int(ARGS.m) > ARGS.n:
        raise argparse.ArgumentError("neighborhood size 'm' cannot be greater than graph size 'n'")

    if not os.path.exists(ARGS.save_dir):
        os.makedirs(ARGS.save_dir)

    all_timeseries, all_cms = run_simulation(simulation, ARGS.instances, ARGS.processes, ARGS.batch_size)

    np.save(os.path.join(ARGS.save_dir, ARGS.prefix+'_timeseries.npy'), all_timeseries)
    np.save(os.path.join(ARGS.save_dir, ARGS.prefix+'_edge.npy'), all_cms)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', type=int, default=10,
                        help='number of particles')
    parser.add_argument('-m', type=str, default=1,
                        help='number of neighbors for each particle')
    parser.add_argument('-k', type=int, default=4,
                        help='size of state vector')
    parser.add_argument('-s', '--steps', type=int, default=50,
                        help='number of timesteps of a trajectory')
    parser.add_argument('-i', '--instances', type=int, default=1,
                        help='number of simulation instances')
    parser.add_argument('--save-dir', type=str,
                        help='name of the save directory')
    parser.add_argument('--prefix', type=str, default='',
                        help='prefix for save files')
    parser.add_argument('--processes', type=int, default=1,
                        help='number of parallel processes')
    parser.add_argument('--batch-size', type=int, default=100,
                        help='number of simulation instances for each process')

    ARGS = parser.parse_args()

    main()
