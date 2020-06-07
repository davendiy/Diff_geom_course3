#!/usr/bin/env python3
# -*-encoding: utf-8-*-

# created: 04.06.2020
# Excusa. Quod scripsi, scripsi.

# by David Zashkolny
# 3 course, comp math
# Taras Shevchenko National University of Kyiv
# email: davendiy@gmail.com

import func as fn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class VectorFunc:

    def __init__(self, x: fn.Function, y: fn.Function, z: fn.Function,
                 t_1: float, t_2: float, str_repr=''):
        assert x.get_vars() - {'t', } == set()
        assert y.get_vars() - {'t', } == set()
        assert z.get_vars() - {'t', } == set()

        self._t = fn.Var('t')
        self._t.right = min(self._t.right, t_2)
        self._t.left = max(self._t.left, t_1)
        self.x = x
        self.y = y
        self.z = z
        self._str_repr = str_repr or f'{x}, {y}, {z}'
        self.left = self._t.left
        self.right = self._t.right

    def dot(self, other) -> fn.Function:
        if not isinstance(other, VectorFunc):
            raise TypeError(f"Bad type for dot: {type(other)}")

        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other):
        if not isinstance(other, VectorFunc):
            raise TypeError(f"Bad type for dot: {type(other)}")

        x = self.y * other.z - self.z * other.y
        y = self.x * other.z - self.z * other.x
        z = self.x * other.y - self.y * other.x
        return VectorFunc(x, y * (-1), z, self.left, self.right,
                          str_repr=f'[{self}, {other}]')

    def norm(self) -> fn.Function:
        return fn.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)

    @classmethod
    def mixed(cls, a, b, c):
        return cls.dot(a, cls.cross(b, c))

    def derivative(self, order=1):
        variables = [self._t] * order
        return VectorFunc(self.x.partial_derivative_n(*variables),
                            self.y.partial_derivative_n(*variables),
                            self.z.partial_derivative_n(*variables),
                            self.left, self.right)

    def __call__(self, t=None):
        if t is not None:
            return np.array([self.x(t=t), self.y(t=t), self.z(t=t)])
        else:
            return np.array([self.x(), self.y(), self.z()])

    def __str__(self):
        return f'({self._str_repr})'

    def __repr__(self):
        return f'VectorFunc({repr(self.x)}, {repr(self.y)}, {repr(self.z)}, ' \
               f'{self.left}, {self.right})'

    def _binary_operation(self, other, operator):
        _f = fn.BINARY_DICT[operator]
        if isinstance(other, VectorFunc):
            x, y, z = _f(self.x, other.x), _f(self.y, other.y), \
                      _f(self.z, other.z)
            left = max(self.left, other.left)
            right = min(self.right, other.right)
            return VectorFunc(x, y, z, left, right)
        elif isinstance(other, fn.Function) or isinstance(other, int) or \
                isinstance(other, float) or isinstance(other, fn.Var):
            x, y, z = _f(self.x, other), _f(self.y, other), _f(self.z, other)
            return VectorFunc(x, y, z, self.left, self.right)

    def __add__(self, other):
        return self._binary_operation(other, fn.ADDITION)

    def __radd__(self, other):
        return self._binary_operation(other, fn.ADDITION)

    def __sub__(self, other):
        return self._binary_operation(other, fn.SUBTRACTION)

    def __rsub__(self, other):
        return self._binary_operation(other, fn.R_SUBTRACTION)

    def __mul__(self, other):
        return self._binary_operation(other, fn.MULTIPLICATION)

    def __rmul__(self, other):
        return self._binary_operation(other, fn.MULTIPLICATION)

    def __truediv__(self, other):
        return self._binary_operation(other, fn.DIVISION)

    def __rtruediv__(self, other):
        return self._binary_operation(other, fn.R_DIVISION)


class Plane:

    def __init__(self, x: fn.Function, y: fn.Function, z: fn.Function,
                 u1=None, u2=None, v1=None, v2=None):
        assert x.get_vars() == {'u', 'v'}
        assert y.get_vars() == {'u', 'v'}
        assert z.get_vars() == {'u', 'v'}
        self.x = x
        self.y = y
        self.z = z
        self._u = fn.Var('u')
        self._v = fn.Var('v')

        self._u.right = min(self._u.right, u2)
        self._u.left = max(self._u.left, u1)
        self._v.right = min(self._v.right, v2)
        self._v.left = max(self._v.left, v1)

    @classmethod
    def from_general(cls, F: fn.Function):
        assert len(F.get_vars()) == 3


    def __str__(self):
        return f"P(u, v) = ({self.x}, {self.y}, {self.z})"

    def show(self):
        pass


class ParamCurve3D:

    def __init__(self, v: VectorFunc):
        self.vector = v

    def derivative(self, order=1):
        return ParamCurve3D(self.vector.derivative(order))

    def __str__(self):
        return f'r(t) = {self.vector}'

    def __repr__(self):
        return f'ParamCurve3D({self.vector.__repr__()})'

    def __call__(self, t=None):
        return self.vector(t)

    def norm(self) -> fn.Function:
        return self.vector.norm()

    def tangent_unit(self):
        """ \tau(t) = r'(t) / |r'(t)|
        """
        dr = self.derivative().vector
        return dr / dr.norm()

    def normal_unit(self):
        """ \nu(t) = [[r'(t), r''(t)], r'(t)] / |[[r'(t), r''(t)], r'(t)]|
        """
        r = self.vector
        dr1 = r.derivative()
        dr2 = r.derivative(order=2)
        res = dr1.cross(dr2).cross(dr1)
        return res / res.norm()

    def binormal_unit(self):
        """\beta(t) = [r'(t), r''(t)] / |[r'(t), r''(t)]|
        """
        r = self.vector
        dr1 = r.derivative()
        dr2 = r.derivative(order=2)
        res = dr1.cross(dr2)
        return res / res.norm()

    # TODO:
    def osculating_plane(self):
        """ (R - r(t0), r'(t0), r''(t0)) = 0
        R = (X, Y, Z)
        """
        r = self.vector
        X = fn.Var('X')
        Y = fn.Var('Y')
        Z = fn.Var('Z')
        dr1 = r.derivative()
        dr2 = dr1.derivative()
        tmp = dr1.cross(dr2)

    # TODO:
    def normal_plane(self):
        """x'(t0)*(X - x(t0)) + y'(t0) * (Y - y(t0)) + z'(t0)*(Z - z(t0)) = 0
        """
        # point = self(t0)
        # dr1 = self.derivative()
        # A, B, C = dr1(t0)
        # D = -(point[0] * A + point[1] * B + point[2] * C)
        # return A, B, C, D

    # TODO:
    def reference_plane(self):
        pass

    def curvature(self) -> fn.Function:
        """k(t) = |[r'(t), r''(t)]| / |r'(t)|^3
        """
        r = self.vector
        dr1 = r.derivative()
        dr2 = r.derivative(order=2)
        return dr1.cross(dr2).norm() / (dr1.norm() ** 3)

    def torsion(self):
        """\\Kappa(t) = (r'(t), r''(t), r'''(t)) / |[r'(t), r''(t)]|^2
        """
        r = self.vector
        dr1 = r.derivative()
        dr2 = r.derivative(order=2)
        dr3 = r.derivative(order=3)
        nom = VectorFunc.mixed(dr1, dr2, dr3)
        denom = dr1.cross(dr2).norm() ** 2
        return nom / denom

    def osculating_circle(self):
        pass

    def reset_axes(self):
        self.ax = plt.figure().add_subplot(111, projection='3d')
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')

        # self.ax.set_xlim(-20, 20)
        # self.ax.set_ylim(-10, 10)
        # self.ax.set_zlim(-10, 10)

    def plot(self, dots=1000, start=-100, end=100):
        self.reset_axes()
        start = max(start, self.vector.left)
        end = min(end, self.vector.right)
        t = np.linspace(start, end, dots)
        x, y, z = self(t)
        self.ax.plot(x, y, z, c='r')

    def frenet(self, t0):
        tau = self.tangent_unit()(t=t0)
        beta = self.binormal_unit()(t=t0)
        nu = self.normal_unit()(t=t0)

        point = self(t=t0)
        x0, y0, z0 = point
        X = [x0, x0, x0]
        Y = [y0, y0, y0]
        Z = [z0, z0, z0]
        return self.ax.quiver(X, Y, Z, tau, beta, nu)


if __name__ == '__main__':

    t = fn.Var('t')
    r = VectorFunc(1 + fn.cos(t), fn.sin(t), 2 * fn.sin(fn.from_var_factory(t) / 2), 0, 4 * np.pi)
    curve = ParamCurve3D(r)
    curve.plot()
    curve.frenet(2)
    plt.show()
