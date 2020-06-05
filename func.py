#!/usr/bin/env python3
# -*-encoding: utf-8-*-

# created: 05.06.2020
# Excusa. Quod scripsi, scripsi.

# by David Zashkolny
# 3 course, comp math
# Taras Shevchenko National University of Kyiv
# email: davendiy@gmail.com

from threading import RLock
import typing as typ
import inspect

ADDITION       = '+'
SUBTRACTION    = '-'
R_SUBTRACTION  = 'r-'
MULTIPLICATION = '*'
DIVISION       = '/'
R_DIVISION     = '\\'
SUPERPOSITION  = '@'
FROM_FUNC      = '_func_doc_'
FROM_VAR       = '_var_name_'
FROM_CONST     = '_constant_'

BINARY = {ADDITION, SUBTRACTION, MULTIPLICATION, DIVISION,
          R_DIVISION, R_SUBTRACTION}

UNARY = {FROM_FUNC, FROM_VAR, FROM_CONST}

SPECIAL = {SUPERPOSITION, }

OPERATIONS = BINARY | UNARY | SPECIAL

BINARY_OPERATIONS = {
    ADDITION:       lambda x, y: x + y,
    SUBTRACTION:    lambda x, y: x - y,
    MULTIPLICATION: lambda x, y: x * y,
    DIVISION:       lambda x, y: x / y,
    R_DIVISION:     lambda x, y: y / x,
    R_SUBTRACTION:  lambda x, y: y - x,
}

LOCK = RLock()


class WTFError(Exception):

    def __str__(self):
        return "Such situation couldn't happen."


class NotInitialisedVarError(TypeError):

    def __init__(self, name):
        super(NotInitialisedVarError, self).__init__()
        self.name = name

    def __str__(self):
        return f"Call for not initialised variable: {self.name}."


class Var:

    __instances = {}

    def __new__(cls, name: str, left=float('-inf'), right=float('inf')):
        if name not in cls.__instances:
            cls.__instances[name] = super().__new__(cls)
        return cls.__instances[name]

    def __init__(self, name: str, left=float('-inf'), right=float('inf')):
        self.name = name
        self._value = None
        self.left = left
        self.right = right

    def set_value(self, value: typ.Union[float, int]):
        with LOCK:
            self._value = value

    def __hash__(self):
        return hash(self.name)

    def __call__(self):
        if self._value is None:
            raise NotInitialisedVarError(self.name)
        return self._value

    def __eq__(self, other):
        if isinstance(other, Var):
            return self.name == other.name
        elif isinstance(other, str):
            return self.name == other
        else:
            return False

    def __add__(self, other):
        return other.__radd__(self)

    def __mul__(self, other):
        return other.__radd__(self)

    def __sub__(self, other):
        return other.__rsup__(self)

    def __truediv__(self, other):
        return other.__rtruediv__(self)

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"Var('{self.name}')"


class Function:

    def __init__(self, main_op: str, variables: typ.Set[Var],
                 *sons: typ.Union[typ.Callable, float, int],
                 name='', **superpos_sons):
        assert main_op in OPERATIONS, 'bad operation'
        assert all(isinstance(var, Var) for var in variables), 'bad variable'
        assert not (len(sons) > 1 and main_op in UNARY), 'bad amount of sons'
        assert not (len(sons) != 2 and main_op in BINARY), 'bad amount of sons'
        assert not (main_op in BINARY and
                    any(not isinstance(son, Function) for son in sons)), 'bad type of sons'

        assert not (len(sons) != 1 and main_op == SUPERPOSITION
                    and len(superpos_sons) == 0 ), 'bad superposition format'

        self._main_op = main_op
        self._vars = variables

        self._superpos_sons = superpos_sons
        self._sons = sons
        self.name = name
        if self._main_op == FROM_FUNC:
            self._check_function()

        if self._main_op == SUPERPOSITION:
            self._check_superposition()

    def _check_superposition(self):
        given_vars = set(self._superpos_sons.keys())
        func = self._sons[0]    # type: Function
        if given_vars != func._vars:
            raise ValueError(f"Bad set of variables are given: "
                             f"expected {func._vars}, got {self._superpos_sons}")

        sons_vars = set()
        for el in self._superpos_sons.values():
            if not isinstance(el, Function):
                raise TypeError(f"Element with type for superposition: {el}")
            sons_vars |= el._vars

        if sons_vars & func._vars:
            raise ValueError(f"Super func should be free from substitutions' "
                             f"variables. Conflicts: {sons_vars & func._vars}")

    def _check_function(self):
        func = self._sons[0]    # type: typ.Callable
        signature = inspect.getfullargspec(func)
        if self._vars != set(signature.args):
            raise ValueError(f"Error checking function signature: "
                             f"expected {self._vars}, got {signature}")

    def _binary(self, other, operation_type):
        assert isinstance(other, Function) or isinstance(other, Var) \
            or isinstance(other, int) or isinstance(other, float), 'bad operand'
        assert operation_type in BINARY, 'bad operation'
        if isinstance(other, Var):
            other = Function(FROM_VAR, {other}, other)
        elif isinstance(other, float) or isinstance(other, int):
            other = Function(FROM_CONST, set(), other)
        return Function(operation_type, self._vars | other._vars,
                        self, other)

    def __add__(self, other):
        return self._binary(other, ADDITION)

    def __sub__(self, other):
        return self._binary(other, SUBTRACTION)

    def __mul__(self, other):
        return self._binary(other, MULTIPLICATION)

    def __truediv__(self, other):
        return self._binary(other, DIVISION)

    def __radd__(self, other):
        return self.__add__(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __rtruediv__(self, other):
        return self._binary(other, R_DIVISION)

    def __rsub__(self, other):
        return self._binary(other, R_SUBTRACTION)

    def superposition(self, **others):
        """ return f(...) = self(other(...))
        """
        given_pars = set(others.keys())
        if self._vars - given_pars:
            raise ValueError(f"Not enough parameters: expected {self._vars}, "
                             f"got {given_pars}")

        res_kwargs = {}
        for var in self._vars:
            value = others[var.name]
            if isinstance(value, Function):
                res_kwargs[var.name] = value
            elif isinstance(value, Var):
                res_kwargs[var.name] = from_var_factory(value)
            elif isinstance(value, int) or isinstance(value, float):
                res_kwargs[var.name] = from_const_factory(value)
            else:
                raise TypeError(f"Element with unknown type for superposition: {value}")

        return Function(SUPERPOSITION, set(), self, name='', **res_kwargs)

    def __call__(self, **kwargs):
        if self._main_op == FROM_VAR:
            if self._sons[0] in kwargs:
                return kwargs[self._sons[0]]
            if not kwargs:
                return self._sons[0]()

        elif self._main_op == FROM_CONST:
            return self._sons[0]

        elif self._main_op == FROM_FUNC:
            given_keys = set(kwargs.keys())
            if not kwargs:
                kwargs = {var.name: var() for var in self._vars}

            elif self._vars - given_keys != set() :
                raise ValueError(f"Not enough parameters: expected {self._vars}, "
                                 f"got {given_keys}")

            res_kwargs = {el.name: kwargs[el.name] for el in self._vars}
            return self._sons[0](**res_kwargs)

        elif self._main_op == SUPERPOSITION:
            res_kwargs = {}
            for var, func in self._superpos_sons.items():
                if isinstance(var, Var):
                    res_kwargs[var.name] = func(**kwargs)
                elif isinstance(var, str):
                    res_kwargs[var] = func(**kwargs)
                else:
                    raise WTFError()
            return self._sons[0](**res_kwargs)

        elif self._main_op in BINARY:
            return BINARY_OPERATIONS[self._main_op](self._sons[0](**kwargs),
                                                    self._sons[1](**kwargs))

        else:
            raise WTFError()


def from_func_factory(func: callable, variables: typ.Set[Var], name=''):
    return Function(FROM_FUNC, variables, func, name=name)


def from_var_factory(var: Var):
    return Function(FROM_VAR, {var}, var)


def from_const_factory(const: typ.Union[int, float]):
    return Function(FROM_CONST, set(), const)
