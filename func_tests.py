#!/usr/bin/env python3
# -*-encoding: utf-8-*-

# created: 05.06.2020
# Excusa. Quod scripsi, scripsi.

# by David Zashkolny
# 3 course, comp math
# Taras Shevchenko National University of Kyiv
# email: davendiy@gmail.com

import unittest
from func import *
import random


# noinspection DuplicatedCode
class MyTestCase(unittest.TestCase):

    def test_vars(self):
        x_var = Var('x')
        x_var.set_value(3)
        self.assertEqual(x_var(), 3)

    def test_SimpleFunc1(self):

        def f1(x1, x2):
            return x1 + x2

        x1_var = Var('x1')
        x2_var = Var('x2')

        test_f = Function(FROM_FUNC, {x1_var, x2_var}, f1, name='f')
        x1_var.set_value(1)
        x2_var.set_value(2)
        self.assertEqual(test_f(), 3)
        self.assertEqual(test_f(x1=1, x2=3, x3=4), 4)

    def test_SimpleFunc2(self):
        x1_var = Var('x1')
        test_f1 = Function(FROM_VAR, {x1_var}, x1_var)
        x1_var.set_value(3)
        self.assertEqual(test_f1(), 3)
        self.assertEqual(test_f1(x1=2, x3=2), 2)

    def test_AddSubFuncs(self):

        def f(x1, x2):
            return x1 + x2

        def g(x1, x2):
            return x1 * x2

        def h(x3):
            return x3 ** 2

        x1_var = Var("x1")
        x2_var = Var("x2")
        x3_var = Var("x3")
        test_f = from_func_factory(f, {x1_var, x2_var})
        test_g = from_func_factory(g, {x1_var, x2_var})
        test_h = from_func_factory(h, {x3_var})

        h1 = test_f + test_g
        h2 = test_f + x1_var + test_h + 2
        h3 = test_f - test_g
        h4 = test_f - x2_var - test_h - 2

        x1_var.set_value(random.randrange(0, 100))
        x2_var.set_value(random.randrange(0, 100))
        x3_var.set_value(random.randrange(0, 100))

        self.assertEqual(h1(), test_f() + test_g())
        self.assertEqual(h2(), test_f() + x1_var() + test_h() + 2)
        self.assertEqual(h3(), test_f() - test_g())
        self.assertEqual(h4(), test_f() - x2_var() - test_h() - 2)

        self.assertEqual(h2(x1=2, x2=3, x3=1),
                         test_f(x1=2, x2=3, x3=1) + 2 + test_h(x1=2, x2=3, x3=1) + 2)

    def test_MulDivFuncs(self):

        def f(x1, x2):
            return x1 + x2

        def g(x1, x2):
            return x1 * x2

        def h(x3):
            return x3 ** 2

        x1_var = Var("x1")
        x2_var = Var("x2")
        x3_var = Var("x3")
        test_f = from_func_factory(f, {x1_var, x2_var})
        test_g = from_func_factory(g, {x1_var, x2_var})
        test_h = from_func_factory(h, {x3_var})

        h1 = test_f * test_g
        h2 = test_f * x1_var * test_h * 2
        h3 = test_f / test_g
        h4 = test_f * (x2_var / test_h) / 2

        x1_var.set_value(random.randrange(0, 100))
        x2_var.set_value(random.randrange(0, 100))
        x3_var.set_value(random.randrange(0, 100))

        self.assertEqual(h1(), test_f() * test_g())
        self.assertEqual(h2(), test_f() * x1_var() * test_h() * 2)
        self.assertEqual(h3(), test_f() / test_g())
        self.assertEqual(h4(), test_f() * (x2_var() / test_h()) / 2)

        self.assertEqual(h2(x1=2, x2=3, x3=1),
                         test_f(x1=2, x2=3, x3=1) * 2 * test_h(x1=2, x2=3, x3=1) * 2)

    def test_superposition(self):
        def f(x1, x2):
            return x1 + x2

        def g(x1, x2):
            return x1 * x2

        def h(x3):
            return x3 ** 2

        def t(x4, x5):
            return x4 - x5

        x1_var = Var('x1')
        x2_var = Var('x2')
        x3_var = Var('x3')
        x4_var = Var('x4')
        x5_var = Var('x5')

        test_f = from_func_factory(f, {x1_var, x2_var})
        test_g = from_func_factory(g, {x1_var, x2_var})
        test_h = from_func_factory(h, {x3_var})
        test_t = from_func_factory(t, {x4_var, x5_var})

        subs_f = test_f.superposition(x1=test_h, x2=test_t)
        subs_g = test_g.superposition(x1=test_h, x2=test_h)

        x1_var.set_value(random.randrange(-100, 100))
        x2_var.set_value(random.randrange(-100, 100))
        x3_var.set_value(random.randrange(-100, 100))
        x4_var.set_value(random.randrange(-100, 100))
        x5_var.set_value(random.randrange(-100, 100))

        self.assertEqual(subs_f(), test_h() + test_t())
        self.assertEqual(subs_g(), test_h() * test_h())

    def test_SimpleDer(self):

        def f(x1, x2, x3):
            return x1 * x2 - x3

        def der_x1(x1, x2, x3):
            return x2

        def der_x2(x1, x2, x3):
            return x1

        def der_x3(x1, x2, x3):
            return -1

        x1_var = Var("x1")
        x2_var = Var("x2")
        x3_var = Var("x3")
        x4_var = Var('x4')

        test_f = from_func_factory(f, {x1_var, x2_var, x3_var})

        f_der_x1 = test_f.partial_derivative(x1_var)
        f_der_x2 = test_f.partial_derivative(x2_var)
        f_der_x3 = test_f.partial_derivative(x3_var)
        f_der_x4 = test_f.partial_derivative(x4_var)

        x1_var.set_value(random.randrange(-100, 100))
        x2_var.set_value(random.randrange(-100, 100))
        x3_var.set_value(random.randrange(-100, 100))
        x4_var.set_value(random.randrange(-100, 100))

        delta = 10e-7
        self.assertAlmostEqual(f_der_x1(), der_x1(x1_var(), x2_var(), x3_var()),
                               delta=delta)
        self.assertAlmostEqual(f_der_x2(), der_x2(x1_var(), x2_var(), x3_var()),
                               delta=delta)
        self.assertAlmostEqual(f_der_x3(), der_x3(x1_var(), x2_var(), x3_var()),
                               delta=delta)
        self.assertEqual(f_der_x4(), 0)

    def test_operatorsDer(self):

        def f(x1, x2, x3):
            return x1 * x2 + x3

        def g(x1, x2):
            return x1 + x2

        x1_var = Var("x1")
        x2_var = Var("x2")
        x3_var = Var("x3")
        x4_var = Var("x4")

        x1_var.set_value(random.randrange(-100, 100))
        x2_var.set_value(random.randrange(-100, 100))
        x3_var.set_value(random.randrange(-100, 100))
        x4_var.set_value(random.randrange(-100, 100))

        test_f = from_func_factory(f, {x1_var, x2_var, x3_var})
        test_g = from_func_factory(g, {x1_var, x2_var})

        delta = 10e-7
        for _x in [x1_var, x2_var, x3_var, x4_var]:
            test_f_der = test_f.partial_derivative(_x)
            test_g_der = test_g.partial_derivative(_x)

            for operator in BINARY_OPERATORS:
                test_func_op = BINARY_DICT[operator](test_f, test_g)  # type: Function
                test_func_op_der = test_func_op.partial_derivative(_x)
                self.assertAlmostEqual(test_func_op_der(),
                                       SIMPLE_DERIVATIVES[operator](test_f(),
                                                                    test_g(),
                                                                    test_f_der(),
                                                                    test_g_der()),
                                       delta=delta)

    def test_superpos(self):

        def f(x1, x2):
            return x1 + x2

        def g(x1, x2):
            return x1 * x2

        def h(x3):
            return x3 ** 2

        def t(x4, x5):
            return x4 - x5

        def f_ht(x3, x4, x5):
            return f(h(x3), t(x4, x5))

        def g_hh(x3):
            return g(h(x3), h(x3))

        x1_var = Var('x1')
        x2_var = Var('x2')
        x3_var = Var('x3')
        x4_var = Var('x4')
        x5_var = Var('x5')

        test_f = from_func_factory(f, {x1_var, x2_var})
        test_g = from_func_factory(g, {x1_var, x2_var})
        test_h = from_func_factory(h, {x3_var})
        test_t = from_func_factory(t, {x4_var, x5_var})

        subs_f = test_f.superposition(x1=test_h, x2=test_t)
        subs_g = test_g.superposition(x1=test_h, x2=test_h)

        subs_f2 = from_func_factory(f_ht, {x3_var, x4_var, x5_var})
        subs_g2 = from_func_factory(g_hh, {x3_var,})

        x1_var.set_value(random.randrange(-100, 100))
        x2_var.set_value(random.randrange(-100, 100))
        x3_var.set_value(random.randrange(-100, 100))
        x4_var.set_value(random.randrange(-100, 100))
        x5_var.set_value(random.randrange(-100, 100))

        delta = 10e-3
        for _x in [x1_var, x2_var, x3_var, x4_var, x5_var]:
            subs_f_der = subs_f.partial_derivative(_x)
            subs_g_der = subs_g.partial_derivative(_x)
            subs_f2_der = subs_f2.partial_derivative(_x)
            subs_g2_der = subs_g2.partial_derivative(_x)
            self.assertAlmostEqual(subs_f_der(), subs_f2_der(), delta=delta)
            self.assertAlmostEqual(subs_g_der(), subs_g2_der(), delta=delta)


if __name__ == '__main__':
    unittest.main()
