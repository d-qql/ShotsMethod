import math
from unittest import TestCase
import numpy as np
import matplotlib.pyplot as plt
from rk_solver import RKSolver, SolverParams


# Не знаешь - не трогай(даже если знаешь - не трогай)
def F(t, current_state):
    x = current_state[0]
    u = current_state[1]
    y = current_state[2]
    nu = current_state[3]
    mu = 0.012277471
    eta = 1 - mu
    A = np.sqrt(((x + mu) ** 2 + y ** 2) ** 3)
    B = np.sqrt(((x - eta) ** 2 + y ** 2) ** 3)
    return np.array([u, x + 2 * nu - eta * (x + mu) / A - mu * (x - eta) / B, nu, y - 2 * u - eta * y / A - mu * y / B])


# Используем метод Дормана Принса
matrix = np.array([[0, 0, 0, 0, 0, 0, 0],
                   [1. / 5, 0, 0, 0, 0, 0, 0],
                   [3. / 40, 9. / 40, 0, 0, 0, 0, 0],
                   [44. / 45, -56. / 15, 32. / 9, 0, 0, 0, 0],
                   [19372. / 6561, -25360. / 2187, 64448. / 6561, -212. / 729, 0, 0, 0],
                   [9017. / 3168, -355. / 33, 46732. / 5247, 49. / 176, -5103. / 18656, 0, 0],
                   [35. / 384, 0, 500. / 1113, 125. / 192, -2187. / 6784, 11. / 84, 0]])
col = np.array([0, 1. / 5, 3. / 10, 4. / 5, 8. / 9, 1, 1])
row = np.array([35. / 384, 0, 500. / 1113, 125. / 192, -2187. / 6784, 11. / 84, 0])
solver = RKSolver(matrix, row, col)

# Укажи шаг по х и точность
params = SolverParams(0.01, 1e-3)
# x, u, y, nu
initial_value = np.array([0.994, 0, 0, -2.00158510637908252240537862224])
ixes, res = solver.solve(F, 0, 170, initial_value, params)
print("t  |             x          |             u             |           y            |               nu          |")
for i in range(len(ixes)):
    print(ixes[i], res[i][0], res[i][1], res[i][2], res[i][3])