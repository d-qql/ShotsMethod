from unittest import TestCase
import numpy as np

"""
Файл с тестами солвера Рунге-Кутты
"""
from rk_solver import RKSolver, SolverParams


def shot():
    """
    y'' = f(x, y, z); z = y'
    """

    # # Указать функцию p(x)
    # def p(x):
    #     return 0
    #
    # # Указать функцию q(x)
    #
    # def q(x):
    #     return -1

    # Указать функцию f(x, y, z)

    def f(x, y, z):
        return y

    # Не знаешь - не трогай(даже если знаешь - не трогай)
    def F(x, y):
        return np.array([y[1], f(x, y[0], y[1])])

    # Используем метод трапеций Рунге-Кутты
    matrix = np.array([[1. / 2, 0], [0, 1. / 2]])
    col = np.array([1. / 2, 1. / 2])
    row = np.array([1. / 2, 1. / 2])
    solver = RKSolver(matrix, row, col)

    # Укажи шаг по х и точность
    params = SolverParams(0.01, 1e-5)

    # Укажи граничное условие на y(x1) и на y'(x2)
    fixed_Y = 1
    x1 = 0
    fixed_dY = 2
    x2 = 1

    # Укажи начальную вариацию параметра, заданного на правой границе (если не сходится - измени его)
    initialVariate = 0

    # Укажи шаг изменения вариации (малый шаг - долго работает)
    variateStep = 0.001

    # Укажи точность нахождения вариации
    # (это точность, с которой мы сравниваем полученное граничное условие на конце в ходе решения с реальным)
    variateTol = 1e-3

    if (x1 < x2):
        start = x1
        end = x2
        VariateDerivative = True
        referenceValue = fixed_dY

    else:
        start = x2
        end = x1
        VariateDerivative = False
        referenceValue = fixed_Y

    if (VariateDerivative):
        ixes, res = solver.solve(F, start, end, np.array([fixed_Y, initialVariate]), params)
    else:
        ixes, res = solver.solve(F, start, end, np.array([initialVariate, fixed_dY]), params)
    firstSign = res[-1][int(VariateDerivative)] - referenceValue > 0
    currentSign = firstSign
    currentVariate = initialVariate
    while currentSign == firstSign:
        currentVariate += variateStep
        if (VariateDerivative):
            ixes, res = solver.solve(F, start, end, np.array([fixed_Y, currentVariate]), params)
        else:
            ixes, res = solver.solve(F, start, end, np.array([currentVariate, fixed_dY]), params)
        currentSign = res[-1][int(VariateDerivative)] - referenceValue > 0

    if (initialVariate < currentVariate):
        leftVariate = initialVariate
        rightVariate = currentVariate
    else:
        leftVariate = currentVariate
        rightVariate = initialVariate

    leftSign = firstSign
    rightSign = currentSign
    while abs(res[-1][int(VariateDerivative)] - referenceValue) > variateTol:
        print(res[-1][int(VariateDerivative)])
        currentVariate = (rightVariate - leftVariate) / 2.
        if (VariateDerivative):
            ixes, res = solver.solve(F, start, end, np.array([fixed_Y, currentVariate]), params)
        else:
            ixes, res = solver.solve(F, start, end, np.array([currentVariate, fixed_dY]), params)
        currentSign = res[-1][int(VariateDerivative)] - referenceValue > 0
        if currentSign != leftSign:
            leftVariate = leftVariate
            rightVariate = currentVariate
            leftSign = leftSign
            rightSign = currentSign
        if (currentSign != rightSign):
            leftVariate = currentVariate
            rightVariate = rightVariate
            leftSign = currentSign
            rightSign = rightSign
    print(ixes)
    print(res)


if __name__ == "__main__":
    shot()
