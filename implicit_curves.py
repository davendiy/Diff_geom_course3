#!/usr/bin/env python3
# -*-encoding: utf-8-*-

# created: 06.06.2020
# Excusa. Quod scripsi, scripsi.

# by David Zashkolny
# 3 course, comp math
# Taras Shevchenko National University of Kyiv
# email: davendiy@gmail.com

import func as fn
import numpy as np
from curves import ParamCurve3D, VectorFunc


class ImplicitFuncTheoremError(TypeError):

    def __init__(self, point):
        self._point = point
        super().__init__(self)

    def __str__(self):
        return f"The curve doesn't satisfy conditions of the implicit theorem " \
               f"at the point: {self._point}."


class NotFoundError(TypeError):

    def __str__(self):
        return f"Something went wrong. Program requires not found value."


class VectorFromImplicit(VectorFunc):

    def __init__(self, x: fn.Function, y: fn.Function, z: fn.Function, t0: float,
                 df: tuple, d2f: tuple, d3f: tuple):

        super(VectorFromImplicit, self).__init__(x, y, z, t0, t0)
        self._df = df
        self._d2f = d2f
        self._d3f = d3f
        self._t0 = t0

    def derivative(self, order=1):
        if order == 1:
            return VectorFunc(fn.from_const_factory(self._df[0]),
                              fn.from_const_factory(self._df[1]),
                              fn.from_const_factory(self._df[2]), self._t0, self._t0)
        elif order == 2:
            return VectorFunc(fn.from_const_factory(self._d2f[0]),
                              fn.from_const_factory(self._d2f[1]),
                              fn.from_const_factory(self._d2f[2]),
                              self._t0, self._t0)
        elif order == 3:
            return VectorFunc(fn.from_const_factory(self._d3f[0]),
                              fn.from_const_factory(self._d3f[1]),
                              fn.from_const_factory(self._d3f[2]), self._t0, self._t0)
        else:
            raise ValueError(f'Can\'t fint such order of derivative: {order}')


class ImplicitCurve3D:

    def __init__(self, F: fn.Function, G: fn.Function):
        assert F.get_vars() - {'x', 'y', 'z'} == set()
        assert G.get_vars() - {'x', 'y', 'z'} == set()
        self._F = F
        self._G = G
        self._vars = self._F.get_vars()
        self.x = fn.Var('x')
        self.y = fn.Var('y')
        self.z = fn.Var('z')

    def implicit_theorem(self, p0, eps=10e-6):
        with fn.LOCK:
            self._precompute_implicit(p0, eps)
            self._der1()
            self._der23()
        return VectorFromImplicit(self._x_func, self._y_func, self._z_func, self._t0,
                                  (self._dxdz, self._dydz, 1),
                                  (self._d2xdz2, self._d2ydz2, 0),
                                  (self._d3xdz3, self._d3ydz3, 0))

    def _get_jacobi(self, xvar) -> fn.Function:
        _vars = self._vars - {xvar, }
        x1, x2 = _vars
        F, G = self._F, self._G
        return F.partial_derivative(x1) * G.partial_derivative(x2) \
                - F.partial_derivative(x2) * G.partial_derivative(x1)

    def _precompute_implicit(self, p0, eps=10e-6):
        x0, y0, z0 = p0
        self._vals = {self.x: x0, self.y: y0, self.z: z0}
        for var in self._vars:
            if abs(self._get_jacobi(var)(x=x0, y=y0, z=z0)) > eps:
                break
        else:
            raise ImplicitFuncTheoremError(p0)
        self._z = var
        self._x, self._y = self._vars - {self._z, }  # type: fn.Var
        self._x_found = {self._vals[self._z]: self._vals[self._x]}
        self._y_found = {self._vals[self._z]: self._vals[self._y]}
        self._t = fn.Var('t')
        self._t0 = self._vals[self._z]
        self._t.set_value(self._t0)

        def _x_func(t):
            if t in self._x_found:
                return self._x_found[t]
            else:
                raise NotFoundError()

        def _y_func(t):
            if t in self._y_found:
                return self._y_found[t]
            else:
                raise NotFoundError()

        self._x_func = fn.from_func_factory(_x_func, {self._t})
        self._y_func = fn.from_func_factory(_y_func, {self._t})
        self._z_func = fn.from_var_factory(self._t)
        self._kwargs1 = {
            self._x.name: fn.from_const_factory(self._vals[self._x]),
            self._y.name: fn.from_const_factory(self._vals[self._y]),
            self._z.name: fn.from_const_factory(self._vals[self._z])
        }
        self._kwargs2 = {
            self._x.name: self._x_func,
            self._y.name: self._y_func,
            self._z.name: self._z_func
        }

        self._dFdx = self._F.partial_derivative(self._x)
        self._dFdy = self._F.partial_derivative(self._y)
        self._dFdz = self._F.partial_derivative(self._z)

        self._dGdx = self._G.partial_derivative(self._x)
        self._dGdy = self._G.partial_derivative(self._y)
        self._dGdz = self._G.partial_derivative(self._z)

        a11, a12 = self._dFdx.substitute(**self._kwargs1)(), \
                   self._dFdy.substitute(**self._kwargs1)()
        a21, a22 = self._dGdx.substitute(**self._kwargs1)(), \
                   self._dGdy.substitute(**self._kwargs1)()
        self._A = np.array([[a11, a12], [a21, a22]])

    def _der1(self):
        dFdx, dFdy, dFdz = self._dFdx, self._dFdy, self._dFdz
        dGdx, dGdy, dGdz = self._dGdx, self._dGdy, self._dGdz

        b1, b2 = dFdz.substitute(**self._kwargs1)(), dGdz.substitute(**self._kwargs1)()

        self._dxdz, self._dydz = np.linalg.solve(self._A, [b1, b2])

        self._add_found(self._dxdz, self._x_found, self._vals[self._x])
        self._add_found(self._dydz, self._y_found, self._vals[self._y])

    def _der23(self):
        dFdx, dFdy, dFdz = self._dFdx.substitute(**self._kwargs2), \
                           self._dFdy.substitute(**self._kwargs2), \
                           self._dFdz.substitute(**self._kwargs2)
        dGdx, dGdy, dGdz = self._dGdx.substitute(**self._kwargs2), \
                           self._dGdy.substitute(**self._kwargs2), \
                           self._dGdz.substitute(**self._kwargs2)

        d2Fdxdt, d2Fdydt, d2Fdzdt = dFdx.partial_derivative(self._t), \
                                    dFdy.partial_derivative(self._t), \
                                    dFdz.partial_derivative(self._t)

        d2Gdxdt, d2Gdydt, d2Gdzdt = dGdx.partial_derivative(self._t), \
                                    dGdy.partial_derivative(self._t), \
                                    dGdz.partial_derivative(self._t)

        b1 = self._dxdz * d2Fdxdt() + self._dydz * d2Fdydt() \
             + d2Fdzdt()
        b2 = self._dxdz * d2Gdxdt() + self._dydz * d2Gdydt() \
             + d2Gdzdt()

        self._d2xdz2, self._d2ydz2 = np.linalg.solve(self._A, [b1, b2])
        self._add_found2(self._d2xdz2, self._x_found, self._vals[self._x])
        self._add_found2(self._d2ydz2, self._y_found, self._vals[self._x])

        d3Fdxdt2, d3Fdydt2, d3Fdzdt2 = d2Fdxdt.partial_derivative(self._t), \
                                       d2Fdydt.partial_derivative(self._t), \
                                       d2Fdzdt.partial_derivative(self._t)

        d3Gdxdt2, d3Gdydt2, d3Gdzdt2 = d2Gdxdt.partial_derivative(self._t), \
                                       d2Gdydt.partial_derivative(self._t), \
                                       d2Gdzdt.partial_derivative(self._t)

        b1 = 2 * self._d2xdz2 * d2Fdxdt() + 2 * self._d2ydz2 * d2Fdydt() \
                + self._dxdz * d3Fdxdt2() + self._dydz * d3Fdydt2() + d3Fdzdt2()
        b2 = 2 * self._d2xdz2 * d2Gdxdt() + 2 * self._d2ydz2 * d2Gdydt() \
                + self._dxdz * d3Gdxdt2() + self._dydz * d3Gdydt2() + d3Gdzdt2()

        self._d3xdz3, self._d3ydz3 = np.linalg.solve(self._A, [b1, b2])

    @staticmethod
    def _add_found(dvdz: float, _v_found: dict, v0: float):
        dvdz = dvdz * fn.Function.delta_x * 2
        v1 = dvdz
        v2 = 0
        _v_found[v0 + fn.Function.delta_x] = v1
        _v_found[v0 - fn.Function.delta_x] = v2

    @staticmethod
    def _add_found2(d2vdz2: float, _v_found: dict, v0: float):
        delta_x = fn.Function.delta_x
        _v_found[v0 + 2 * delta_x] = _v_found[v0 - 2 * delta_x] \
                = 2 * delta_x ** 2 * d2vdz2 + _v_found[v0]


if __name__ == "__main__":
    x_func = fn.from_var_factory(fn.Var('x'))
    y_func = fn.from_var_factory(fn.Var('y'))
    z_func = fn.from_var_factory(fn.Var('z'))

    test_F = x_func + fn.sinh(x_func) - fn.sin(y_func) - y_func + z_func * 0
    test_G = z_func + fn.exp(z_func) - x_func - fn.log(1 + x_func) - 1 + y_func * 0

    p = (0, 0, 0)

    test = ImplicitCurve3D(test_F, test_G)
    res = test.implicit_theorem(p)

    curve = ParamCurve3D(res)

    print(res(1))
    print("r':", res.derivative()(1))
    print("r'':", res.derivative(order=2)(1))
    print("r''':", res.derivative(order=3)(1))

    print("nu:", curve.normal_unit()())
    print("tau:", curve.tangent_unit()())
    print("beta:", curve.binormal_unit()())

    print("curvature", curve.curvature()())
    print("torsion", curve.torsion()())
