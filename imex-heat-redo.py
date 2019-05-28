import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sparse

from mpl_toolkits.mplot3d import Axes3D
from scipy.sparse.linalg import spsolve

dx = 0.001
dt = 0.001
x = np.arange(0, 1, dx)
t = np.arange(0, 1, dt)
n_x = len(x)
n_t = len(t)

u = np.zeros([n_x, n_t])
u[:, 0] = np.sin(np.pi * x)

# use csr to avoid spsolve() coercing to csr
implicit_matrix = sparse.diags(
    diagonals=[
        np.full(n_x - 1, dt / (2 * dx) - dt / (dx**2)),
        np.full(n_x, 1 + 2 * dt / (dx**2)),
        np.full(n_x - 1, -dt / (2 * dx) - dt / (dx**2)),
    ],
    offsets=[-1, 0, 1],
    format="csr"
)

start_time = time.time()
u[:, 1] = spsolve(implicit_matrix, u[:, 0])
time_solve = time.time() - start_time
print(f"{n_x} by {n_x} system took {time_solve} s to solve")

imex_lhs_matrix = sparse.diags(
    diagonals=[
        np.full(n_x - 1, -dt / (2 * dx**2)),
        np.full(n_x, 1 + dt / (dx ** 2)),
        np.full(n_x - 1, -dt / (2 * dx**2))
    ],
    offsets=[-1, 0, 1],
    format="csr"
)
imex_rhs_curr = sparse.diags(
    diagonals=[
        np.full(n_x - 1, - 3 * dt / (4 * dx) + dt / (2 * dx**2)),
        np.full(n_x, 1 - dt / (dx**2)),
        np.full(n_x - 1, 3 * dt / (4 * dx) + dt / (2 * dx**2)),
    ],
    offsets=[-1, 0, 1],
    format="csr"
)
imex_rhs_prev = sparse.diags(
    diagonals=[
        np.full(n_x - 1, dt / (4 * dx)),
        np.full(n_x - 1, -dt / (4 * dx)),
    ],
    offsets=[-1, 1],
    format="csr"
)

start_time = time.time()
for i in range(2, n_t):
    u[:, i] = spsolve(
        imex_lhs_matrix,
        imex_rhs_curr @ u[:, i - 1] + imex_rhs_prev @ u[:, i - 2]
    )

time_imex = time.time() - start_time
print(f"{n_x} by {n_t} grid took {time_imex} s to evaluate")

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x_grid, t_grid = np.meshgrid(x, t)
ax.plot_surface(t_grid, x_grid, u.transpose())
plt.show()
