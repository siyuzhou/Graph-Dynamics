import numpy as np

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
