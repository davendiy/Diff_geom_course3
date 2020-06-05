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


class MyTestCase(unittest.TestCase):

    def test_vars(self):
        x = Var('x')
        x.set_value(3)
        self.assertEqual(x(), 3)

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

        x1_var.set_value(1)
        x2_var.set_value(3)
        x3_var.set_value(4)

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

        x1_var.set_value(1)
        x2_var.set_value(3)
        x3_var.set_value(4)

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

        x1_var.set_value(1)
        x2_var.set_value(2)
        x3_var.set_value(4)
        x4_var.set_value(5)
        x5_var.set_value(2)

        self.assertEqual(subs_f(), test_h() + test_t())
        self.assertEqual(subs_g(), test_h() * test_h())


if __name__ == '__main__':
    unittest.main()
