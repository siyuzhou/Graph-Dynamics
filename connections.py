import numpy as np


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
