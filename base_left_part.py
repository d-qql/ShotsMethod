from abc import ABC, abstractmethod
"""
Абстрактный класс функции f(x), где x' = f(x)
"""
from mytypes import Vector


class BaseLeftPart(ABC):

    @abstractmethod
    def __call__(self, time: float, values: Vector):
        """

        :param time: Время
        :param values: Вектор значений
        :return: Вектор производных в момент времени time
        """
