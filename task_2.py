import math
from unittest import TestCase
import numpy as np

from rk_solver import RKSolver, SolverParams


def f(x, y, z):
    return -y + 1000 * (1 - z * z) * z


# Не знаешь - не трогай(даже если знаешь - не трогай)
def F(x, y):
    return np.array([y[1], f(x, y[0], y[1])])


# Используем метод трапеций Рунге-Кутты
matrix = np.array([[1. / 2, 0], [0, 1. / 2]])
col = np.array([1. / 2, 1. / 2])
row = np.array([1. / 2, 1. / 2])
solver = RKSolver(matrix, row, col)

# Укажи шаг по х и точность
params = SolverParams(0.001, 1e-3)
fixed_Y = 0
t0 = 0
fixed_dY = 0.001
ixes, res = solver.solve(F, t0, 1000, np.array([fixed_Y, fixed_dY]), params)
