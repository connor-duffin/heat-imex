import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import time

start_time = time.time()
dx = 0.01
dt = 0.001
x = np.arange(0, 1, dx)
n_x = len(x)
t = np.arange(0, 1, dt)
n_t = len(t)

u = np.zeros([n_x, n_t])
u[:, 0] = np.sin(np.pi * x)

explicit_matrix = sparse.diags(
    diagonals=[
        np.full(n_x - 1, - dt / (2 * dx) + dt / (dx**2)),
        np.full(n_x, - 2 * dt / (dx**2)),
        np.full(n_x - 1, dt / (2 * dx) + dt / (dx**2)),
    ],
    offsets=[-1, 0, 1],
    format="csc"
)

u[:, 1] = u[:, 0] + explicit_matrix @ u[:, 0]

imex_lhs_matrix = sparse.diags(
    diagonals=[
        np.full(n_x - 1, -dt / (2 * dx**2)),
        np.full(n_x, 1 + dt / (dx ** 2)),
        np.full(n_x - 1, -dt / (2 * dx**2))
    ],
    offsets=[-1, 0, 1],
    format="csc"
)
imex_rhs_curr = sparse.diags(
    diagonals=[
        np.full(n_x - 1, - 3 * dt / (4 * dx) + dt / (2 * dx**2)),
        np.full(n_x, 1 - dt / (dx**2)),
        np.full(n_x - 1, 3 * dt / (4 * dx) + dt / (2 * dx**2)),
    ],
    offsets=[-1, 0, 1],
    format="csc"
)
imex_rhs_prev = sparse.diags(
    diagonals=[
        np.full(n_x - 1, dt / (4 * dx)),
        np.full(n_x - 1, -dt / (4 * dx)),
    ],
    offsets=[-1, 1],
    format="csc"
)

for i in range(2, n_t):
    u[:, i] = spsolve(
        imex_lhs_matrix,
        imex_rhs_curr @ u[:, i - 1] + imex_rhs_prev @ u[:, i - 2]
    )

total_time = time.time() - start_time
print(f"A {n_x} by {n_t} grid took {total_time} seconds to evaluate")

plt.plot(u[:, 0])
plt.plot(u[:, 1])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x_grid, t_grid = np.meshgrid(x, t)
ax.plot_surface(t_grid, x_grid, u.transpose(), color='b')
plt.show()
