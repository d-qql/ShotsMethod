"""
Класс методов Рунге-Кутты решения СНЛДУ
"""
import copy
import numpy as np
from base_solver import BaseSolver, SolverParams
from mytypes import Vector, Matrix
from base_left_part import BaseLeftPart


class RKSolver(BaseSolver):
    """
    Класс методов Рунге-Кутты
    """

    def __init__(self, rk_matrix: Matrix, weights: Vector, nodes: Vector):
        """

        :param rk_matrix: Матрица Рунге-Кутты
        :param weights: Строка коэффициентов
        :param nodes: Столбец коэффициентов
        """
        self.rk_matrix = rk_matrix
        self.weights = weights
        self.nodes = nodes

    def calc_one_step(
            self,
            left_part: BaseLeftPart,
            current_x: float,
            step: float,
            current_state: Vector,
            step_solution_tol: float,
            max_steps: int
    ):
        """
        Расчет значения функции через шаг по времени длиной step в момент current_time
        со значением y(current_time) = current_state
        :param max_steps: Максимально допустимое количество шагов метода Гаусса-Зейделя
        :param left_part: f(x,t), где x' = f(x, t)
        :param current_x: Текущий x
        :param step: Шаг по x
        :param current_state: Известное значение функции в текущий момент времени
        :param step_solution_tol: Точность решения системы, определяющей векторы k
        :return: Значение функции в новый момент времени
        """
        delta_k = 0
        k = np.zeros((len(current_state), len(self.rk_matrix)))  # Matrix contains vectors "k" in cols
        for steps in range(max_steps):
            delta_k = 0
            for i in range(len(self.rk_matrix)):
                new_k = left_part(current_x + self.nodes[i] * step,
                                  current_state + step * np.dot(k, self.rk_matrix[i, :]))
                delta_k = max(np.linalg.norm(new_k - k[:, i]), delta_k)
                k[:, i] = new_k
            if delta_k < step_solution_tol:
                break
        delta_y = step * np.dot(k, self.weights)
        return current_state + delta_y, delta_k < step_solution_tol

    def solve(self, left_part: BaseLeftPart, initial_x: float, max_x: float, initial_state: Vector, params: SolverParams):
        """
        Вычисление итераций с фиксированным шагом пока не достигнуто максимальное значение времени
        :param params:
        :param max_x: Максимальное x
        :param initial_x: Начальное x
        :param left_part: f(x,t), где x' = f(x, t)
        :param initial_state: Начальное значение
        :return: Массив времен и массив значений в соответствующие моменты времени
        """

        t = initial_x
        y = initial_state

        times = []
        values = []
        times.append(t)
        values.append(copy.deepcopy(y))
        step = params.step
        while t < max_x:
            state, flag = self.calc_one_step(left_part, t, params.step, y, params.step_solution_tol, params.max_steps)
            if flag:
                y = state
                t += params.step
                times.append(t)
                values.append(copy.deepcopy(y))
                step = params.step
            else:
                step /= 2
        return np.array(times), np.array(values)

    def solve_with_tol(self, left_part: BaseLeftPart, initial_time: float, max_time: float, initial_state: Vector,
                       params: SolverParams):
        """
        Вычисление итераций с адаптивным шагом по времени и оценкой ошибки решения
        :param params:
        :param left_part: f(x,t), где x' = f(x, t)
        :param initial_time: Начальное время
        :param max_time: Максимальное время расчета
        :param initial_state: Начальное значение
        :return: Массив времен, массив соответствующих значений и массив оценок ошибок
        """
        step = params.step
        t = initial_time
        y = initial_state
        times = []
        values = []
        tolerance = []
        times.append(t)
        values.append(copy.deepcopy(initial_state))
        while t < max_time:
            y2n, flag1 = self.calc_one_step(left_part, t, step / 2, y, params.step_solution_tol, params.max_steps)
            y2n, flag2 = self.calc_one_step(left_part, t + step / 2, step / 2, y2n, params.step_solution_tol, params.max_steps)
            yn, flag = self.calc_one_step(left_part, t, step, y, params.step_solution_tol, params.max_steps)
            if flag:
                eps = np.linalg.norm(yn - y2n) / (1 - 0.5 ** len(self.rk_matrix))
                if eps != 0:
                    new_step = step * (params.solution_tol / eps) ** 1. / (len(self.rk_matrix) + 1)
                    if new_step > step * 3:
                        step *= 3
                    else:
                        step = new_step
                y, flag0 = self.calc_one_step(left_part, t, step, y, params.step_solution_tol, params.max_steps)
                t += step
                times.append(t)
                values.append(copy.deepcopy(y))
                tolerance.append(eps)
                step = params.step
            else:
                step /= 2
        return np.array(times), np.array(values), np.array(tolerance)
