from abc import ABC, abstractmethod

"""
Базовый класс солвера СНЛДУ
"""
from mytypes import Vector


class SolverParams:
    def __init__(self, step: float, step_solution_tol: float, solution_tol=0, max_steps=50):
        self.step = step  # Шаг по x
        self.step_solution_tol = step_solution_tol  # Точность итерационного процесса вычисления одного шага
        self.solution_tol = solution_tol  # Точность решения системы (используется для динамического шага по времени)
        self.max_steps = max_steps # Максимально допустимое количество шагов


class BaseSolver(ABC):
    """
    Абстрактный класс солвера СНЛДУ
    """

    @abstractmethod
    def solve(self, left_part, initial_x: float, max_x: float, initial_state: Vector, params: SolverParams):
        """

        :param params:
        :param max_x: Максимальное время расчета
        :param initial_x: Начальное время
        :param left_part: f(x,t), где x' = f(x, t)
        :param initial_state: Начальное значение
        :return: Массив времен и массив значений в соответствующие моменты времени
        """
