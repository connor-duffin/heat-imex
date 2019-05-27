import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import time

start_time = time.time()
dx = 0.01
dt = 0.00001
x = np.arange(0, 1, dx)
n_x = len(x)
t = np.arange(0, 1, dt)
n_t = len(t)

u = np.zeros([n_x, n_t])
u[:, 0] = np.sin(np.pi * x)
u[:, 1] = np.sin(np.pi * x)

imex_lhs_matrix = sparse.diags(
    [np.full(n_x - 1, 1 / (4*dx)), 1 / dt, np.full(n_x - 1, -1 / (4*dx))],
    [-1, 0, 1],
    format="csc"
)
imex_rhs_first = sparse.diags([
       np.full(n_x - 1, 3 / (2 * dx**2) + 1 / (4 * dx)),
       -3 * dt / (dx**2) + 1,
       3 * dt / (2 * dx**2) - 1 / (4 * dx)
    ],
    [-1, 0, 1]
)
imex_rhs_second = sparse.diags(
    [np.full(n_x - 1, - 1 / (2 * dx**2)), 1 / (dx**2), -1 / (2 * dx**2)],
    [-1, 0, 1]
)

for i in range(2, n_t):
    u[:, i] = spsolve(
        imex_lhs_matrix,
        imex_rhs_first @ u[:, i - 1] + imex_rhs_second @ u[:, i - 2]
    )


from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x_grid, t_grid = np.meshgrid(x, t)
ax.plot_surface(t_grid, x_grid, u.transpose(), color='b')
plt.show()
