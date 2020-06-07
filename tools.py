#!/usr/bin/env python3
# -*-encoding: utf-8-*-

# created: 04.06.2020
# Excusa. Quod scripsi, scripsi.

# by David Zashkolny
# 3 course, comp math
# Taras Shevchenko National University of Kyiv
# email: davendiy@gmail.com

import inspect
from typing import Union, Iterable, Tuple
import numpy as np
from copy import deepcopy
from collections.abc import Iterable as _IterableType


def is_iterable(x):
    return isinstance(x, _IterableType)


X_TYPE = Union[float, Iterable[float],
               np.ndarray, Iterable[np.ndarray]]


class Function:

    delta_x = 10e-5

    def __init__(self, func: callable, variables=(), str_repr='',
                 superposition=False):
        if isinstance(func, Function):
            variables = variables or func._vars
            if not str_repr:
                res = func._str_repr
                for old_var, new_var in variables:
                    res = res.replace(old_var, new_var)
                str_repr = res

            func = func.func

        self._str_repr = str_repr or func.__doc__ or 'Error no str repr'
        self._superposition = superposition
        self.func = func
        self._vars_num = 1
        if variables is not None:
            for el in variables:
                if not isinstance(el, str):
                    raise ValueError(f"Bad variable type: {el}")
            self._vars_num = len(variables)
            self._vars = variables
        else:
            self._vars = None
            self._create_variables()

    def __str__(self):
        return f"Function({self._str_repr})"

    def __repr__(self):
        return str(self)

    def _create_variables(self):
        n = len(inspect.getfullargspec(self.func).args)
        self._vars_num = n
        self._vars = [f'x{i}' for i in range(1, n+1)]

    def variables(self):
        return self._vars

    def partial_derivative(self, variable: str):

        if variable not in self._vars:
            raise ValueError(f"Bad name of variable: {variable}.")

        i = self._vars.index(variable)

        def _res_func(*x):
            x1 = list(deepcopy(x))
            x1[i] += self.delta_x
            f1 = self(*x1)

            x1[i] -= 2 * self.delta_x
            f2 = self(*x1)
            return (f1 - f2) / (self.delta_x * 2)

        return Function(_res_func, self._vars,
                        str_repr=f'd/d{variable} ( {self} )')

    def _partial_derivative_n(self, n: int, variables: Tuple[str]):

        if n < 1:
            raise ValueError(f'Bad derivative power value: {n}.')

        if n == 1:
            return self.partial_derivative(variables[0])
        else:
            tmp = self._partial_derivative_n(n-1, variables[1:])
            return tmp.partial_derivative(variables[0])

    def partial_derivative_n(self, n: int, variables: Iterable[str]):
        return self._partial_derivative_n(n, tuple(sorted(variables)))

    def __call__(self, *args: Union[float, np.ndarray]) \
            -> Union[float, np.ndarray]:
        if len(args) != self._vars_num:
            raise ValueError(f"Bad amount of variables: expected {self._vars_num}, got {len(args)}.")

        return self.func(*args)

    def norm(self):

        def res_f(*args):
            values = self(*args)
            if is_iterable(values) and not is_iterable(args[0]):
                res = np.sum(values**2, axis=0)
                res = np.sqrt(res)
            elif is_iterable(values) and values.ndim > 1:
                res = np.sum(values**2, axis=0)
                res = np.sqrt(res)
            else:
                res = np.abs(values)
            return res

        return Function(res_f, variables=self._vars, str_repr=f'|{self}|')


if __name__ == '__main__':
    def f(x):
        """ x**2 """
        return x**2

    def g(x, y, z):
        """ x**2 + y**2 + z**2, x * y """
        return np.array((x**2 + y**2 + z**2, x * y))

    test1 = Function(f, ['x'])
    test2 = Function(g, ['x', 'y', 'z'])

    test1_der = test1.partial_derivative_n(2, ['x', 'x'])
    test2_der = test2.partial_derivative('y')
    print(test1_der(1), test2_der(1, 2, 3))

    print(test1_der(np.array([1, 2, 3, 4.4, 5])))
    print(test2_der(np.array([1., 1, 2]), np.array([2., 2, 2]),
                    np.array([3., 3, 3])))
