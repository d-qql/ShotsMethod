import math
from unittest import TestCase
import numpy as np

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
        return np.sqrt(1. / x ** 2 + np.e / np.log(x) * y ** 2 - np.exp(z) * y)

        # Не знаешь - не трогай(даже если знаешь - не трогай) (вектор неизвестных системы ОДУ)

    def F(x, y):
        return np.array([y[1], f(x, y[0], y[1])])

    # Используем метод трапеций Рунге-Кутты
    matrix = np.array([[1. / 2, 0], [0, 1. / 2]])
    col = np.array([1. / 2, 1. / 2])
    row = np.array([1. / 2, 1. / 2])
    solver = RKSolver(matrix, row, col)

    # Укажи шаг по х и точность
    params = SolverParams(0.001, 1e-3)

    # Укажи граничное условие на y(x1) и на y'(x2)
    fixed_Y1 = math.e
    x1 = math.e
    fixed_Y2 = 2 * math.e ** 2
    x2 = math.e ** 2
    referenceValue = fixed_Y2

    # Укажи начальную начальную вариацию параметра
    initialVariate = 1.98
    # Укажи шаг изменения вариации (малый шаг - долго работает)
    variateStep = 0.01

    # Укажи точность нахождения вариации
    # (это точность, с которой мы сравниваем полученное граничное условие на конце в ходе решения с реальным)
    variateTol = 1e-3

    ixes, res = solver.solve(F, x1, x2, np.array([fixed_Y1, initialVariate]), params)

    firstSign = res[-1][0] - referenceValue > 0
    currentSign = firstSign
    currentVariate = initialVariate
    while currentSign == firstSign:
        currentVariate += variateStep
        ixes, res = solver.solve(F, x1, x2, np.array([fixed_Y1, currentVariate]), params)
        currentSign = res[-1][0] - referenceValue > 0

    if (initialVariate < currentVariate):
        leftVariate = initialVariate
        rightVariate = currentVariate
    else:
        leftVariate = currentVariate
        rightVariate = initialVariate

    leftSign = firstSign
    rightSign = currentSign
    while abs(res[-1][0] - referenceValue) > variateTol:
        print("var: ", currentVariate)
        print(res[-1][0])
        currentVariate = (rightVariate + leftVariate) / 2.
        ixes, res = solver.solve(F, x1, x2, np.array([fixed_Y1, currentVariate]), params)
        currentSign = res[-1][0] - referenceValue > 0
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
    # print(ixes)
    # print(res)
    print("https://github.com/d-qql/ShotsMethod")
    print("Начальное приближение вариации отбрасыванием косинуса получено аналитически = -7/6, округлено до -1.5")
    print("x  |             y          |             y'             |")
    for i in range(len(ixes)):
        print(ixes[i], res[i][0], res[i][1])
    return ixes, res



if __name__ == "__main__":
    shot()