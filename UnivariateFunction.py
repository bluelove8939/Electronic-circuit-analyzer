import os
import math
from typing import List
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp


current_dir = os.path.abspath(os.path.split(__file__)[0])

# mathmetical constants
E = math.e
PI = math.pi
INF = float('inf')
NAN = float('nan')
MAX = 'MAX'
STD_EPS = 0.001
MIN_EPS = 0.00001


class Variable(object):
    def __init__(self, value=None, name='default'):
        self._value, self._name = None, None

        self.set(value)
        self.setname(name)

    def cpy(self):
        ret = Variable(value=self._value, name=self._name)
        return ret

    def set(self, value):
        if self._value is not None:
            if isinstance(self._value, (int, float)):
                self._value = float(self._value)
            else:
                raise TypeError("'value' needs to be int or float")
        else:
            self._value = value

        return self._value

    def get(self):
        return self._value

    def empty(self):
        return self._value is None

    def setname(self, name: str):
        self._name = name

    def name(self):
        return self._name

    def __eq__(self, other):
        if not isinstance(other, Variable):
            raise TypeError("'Variable' object does not support item assignment")
        else:
            self._value = other._value

    def __add__(self, other):
        if isinstance(other, Variable):
            item = other.get()
        else:
            item = other

        return Variable(self._value + item)

    def __sub__(self, other):
        if isinstance(other, Variable):
            item = other.get()
        else:
            item = other

        return Variable(self._value - item)

    def __mul__(self, other):
        if isinstance(other, Variable):
            item = other.get()
        else:
            item = other

        return Variable(self._value * item)

    def __div__(self, other):
        if isinstance(other, Variable):
            item = other.get()
        else:
            item = other

        return Variable(self._value / item)

    def __mod__(self, other):
        if isinstance(other, Variable):
            item = other.get()
        else:
            item = other

        return Variable(self._value % item)

    def __divmod__(self, other):
        if isinstance(other, Variable):
            item = other.get()
        else:
            item = other

        return Variable(self._value / item), Variable(self._value % item)

    def __truediv__(self, other):
        if isinstance(other, Variable):
            item = other.get()
        else:
            item = other

        return Variable(self._value / item)

    def __floordiv__(self, other):
        if isinstance(other, Variable):
            item = other.get()
        else:
            item = other

        return Variable(self._value // item)

    def __neg__(self):
        return self * (-1)

    def __pow__(self, power, modulo=None):
        if isinstance(power, Variable):
            item = power.get()
        else:
            item = power

        return Variable(self._value.__pow__(item, modulo))

    def __str__(self):
        return str(self._value)

    def log(self, base):
        if isinstance(base, Variable):
            item = base.get()
        else:
            item = base

        return Variable(math.log(self._value, item))

    def sin(self):
        return Variable(math.sin(self._value))

    def cos(self):
        return Variable(math.cos(self._value))


class FunctionNode(object):
    def __init__(self, lnksize: int, resizable=False, eps=STD_EPS):
        self._lnksize = lnksize
        self._lnkpivot = 0
        self.resizable = resizable
        self.link: List[FunctionNode or None] = [None for _ in range(self._lnksize)]
        if isinstance(eps, (int, float)):
            self.eps = float(eps)
        else:
            raise TypeError("'eps' needs to be int or float")

    def set_eps(self, eps=0.001):
        for lnk in self.link:
            if lnk is not None:
                lnk.set_eps(eps)

    def clear(self):
        self.link = [None for _ in range(self._lnksize)]

    def pack(self, nd, lnk=0):
        if not isinstance(lnk, int):
            raise TypeError("'lnk' needs to be int")

        if not isinstance(nd, FunctionNode):
            raise TypeError("'nd' needs to be 'FunctionNode'")

        if lnk not in range(self._lnksize):
            raise TypeError(f"'lnk' out of range({self._lnksize})")

        self.link[lnk] = nd

        return self

    def append(self, nd):
        if not isinstance(nd, FunctionNode):
            raise TypeError("'nd' needs to be 'FunctionNode'")

        if (self._lnkpivot >= self._lnksize) and not self.resizable:
            raise TypeError("cannot use 'append' method because 'FunctionNode' is not resizable")
        elif self._lnkpivot > self._lnksize:
            raise TypeError("error occurred(_lnkpivot is greater than _lnksize)")
        elif self._lnkpivot == self._lnksize:
            self._lnkpivot += 1
            self._lnksize = self._lnkpivot
            self.link.append(nd)
        else:
            self.link[self._lnkpivot] = nd
            self._lnkpivot += 1

        return self

    def remove(self, lnk: (int, float)):
        if lnk in range(self._lnksize):
            self.link[lnk] = None
        else:
            raise TypeError(f"'lnk' out of range({self._lnksize})")

        return self

    def isfull(self) -> bool:
        for item in self.link:
            if item is None:
                return False
        return True

    def getlink(self, lnk=0):
        if not isinstance(lnk, int):
            raise TypeError("'lnk' needs to be int")

        if lnk in range(self._lnksize):
            return self.link[lnk]
        else:
            raise TypeError(f"'lnk' out of range({self._lnksize})")

    def calculate(self, arg: Variable) -> Variable:
        pass

    def cpy(self):
        pass

    def derivative(self):
        pass

    def expression(self, symbol):
        pass

    def __add__(self, other):
        ret = AddNode()
        if isinstance(other, FunctionNode):
            ret.append(self.cpy()).append(other.cpy())
        elif isinstance(other, (int, float)):
            ret.append(self.cpy()).append(ScalarNode(other))
        elif isinstance(other, Variable):
            ret.append(self.cpy()).append(ScalarNode(other.get()))
        else:
            raise TypeError("invalid 'nd'")
        return ret

    def __sub__(self, other):
        ret = SubNode()
        if isinstance(other, FunctionNode):
            ret.append(self.cpy()).append(other.cpy())
        elif isinstance(other, (int, float)):
            ret.append(self.cpy()).append(ScalarNode(other))
        elif isinstance(other, Variable):
            ret.append(self.cpy()).append(ScalarNode(other.get()))
        else:
            raise TypeError("invalid 'nd'")
        return ret

    def __mul__(self, other):
        ret = MulNode()
        if isinstance(other, FunctionNode):
            ret.append(self.cpy()).append(other.cpy())
        elif isinstance(other, (int, float)):
            ret.append(self.cpy()).append(ScalarNode(other))
        elif isinstance(other, Variable):
            ret.append(self.cpy()).append(ScalarNode(other.get()))
        else:
            raise TypeError("invalid 'nd'")
        return ret

    def __div__(self, other):
        ret = DivNode()
        if isinstance(other, FunctionNode):
            ret.append(self.cpy()).append(other.cpy())
        elif isinstance(other, (int, float)):
            ret.append(self.cpy()).append(ScalarNode(other))
        elif isinstance(other, Variable):
            ret.append(self.cpy()).append(ScalarNode(other.get()))
        else:
            raise TypeError("invalid 'nd'")
        return ret

    def __radd__(self, other):
        return self.__add__(other)

    def __rsub__(self, other):
        ret = SubNode()
        if isinstance(other, FunctionNode):
            ret.append(other.cpy()).append(self.cpy())
        elif isinstance(other, (int, float)):
            ret.append(ScalarNode(other)).append(self.cpy())
        elif isinstance(other, Variable):
            ret.append(ScalarNode(other.get())).append(self.cpy())
        else:
            raise TypeError("invalid 'nd'")
        return ret

    def __rmul__(self, other):
        return self.__mul__(other)

    def __rdiv__(self, other):
        ret = DivNode()
        if isinstance(other, FunctionNode):
            ret.append(other.cpy()).append(self.cpy())
        elif isinstance(other, (int, float)):
            ret.append(ScalarNode(other)).append(self.cpy())
        elif isinstance(other, Variable):
            ret.append(ScalarNode(other.get())).append(self.cpy())
        else:
            raise TypeError("invalid 'nd'")
        return ret

    def __truediv__(self, other):
        return self.__div__(other)

    def __floordiv__(self, other):
        return self.__div__(other)

    def __rtruediv__(self, other):
        return self.__rdiv__(other)

    def __rfloordiv__(self, other):
        return self.__rdiv__(other)

    def __neg__(self):
        ret = NegNode().append(self.cpy())
        return ret

    def __pow__(self, power, modulo=None):
        ret = SqrNode()
        if isinstance(power, ScalarNode):
            ret.value = power.value
            ret.append(self.cpy())
        elif isinstance(power, (int, float)):
            ret.value = power
            ret.append(self.cpy())
        elif isinstance(power, Variable):
            ret.value = power.get()
            ret.append(self.cpy())
        else:
            raise TypeError("invalid 'nd'")
        return ret

    def __rpow__(self, power, modulo=None):
        ret = ExpNode()
        if isinstance(power, ScalarNode):
            ret.value = power.value
            ret.append(self.cpy())
        elif isinstance(power, (int, float)):
            ret.value = power
            ret.append(self.cpy())
        elif isinstance(power, Variable):
            ret.value = power.get()
            ret.append(self.cpy())
        else:
            raise TypeError("invalid 'nd'")
        return ret


class VarNode(FunctionNode):
    def __init__(self):
        super(VarNode, self).__init__(lnksize=0, resizable=False)

    def calculate(self, arg: Variable) -> Variable:
        return Variable(arg.get())

    def cpy(self) -> FunctionNode:
        return VarNode()

    def derivative(self) -> FunctionNode:
        return ScalarNode(value=1)

    def expression(self, symbol):
        return symbol


class ScalarNode(FunctionNode):
    def __init__(self, value: (float, int)):
        super(ScalarNode, self).__init__(lnksize=0, resizable=False)
        self.value = float(value)

    def calculate(self, arg: Variable):
        return Variable(self.value)

    def cpy(self) -> FunctionNode:
        return ScalarNode(value=self.value)

    def derivative(self) -> FunctionNode:
        return ScalarNode(value=0)

    def expression(self, symbol):
        return self.value


class AddNode(FunctionNode):
    def __init__(self, lnksize=0, resizable=True):
        super(AddNode, self).__init__(lnksize, resizable)

    def calculate(self, arg: Variable) -> Variable:
        ret = Variable(0)
        for lnk in self.link:
            ret += lnk.calculate(arg)
        return ret

    def cpy(self) -> FunctionNode:
        root = AddNode()
        for lnk in self.link:
            if lnk is not None:
                root.append(lnk.cpy())
        return root

    def derivative(self) -> FunctionNode:
        root = AddNode()
        for lnk in self.link:
            root.append(lnk.derivative())
        return root

    def expression(self, symbol):
        ret = 0
        for lnk in self.link:
            ret += lnk.expression(symbol)
        return ret


class SubNode(FunctionNode):
    def __init__(self):
        super(SubNode, self).__init__(lnksize=2)

    def calculate(self, arg: Variable) -> Variable:
        return self.link[0].calculate(arg) - self.link[1].calculate(arg)

    def cpy(self) -> FunctionNode:
        root = SubNode()
        for lnk in self.link:
            if lnk is not None:
                root.append(lnk.cpy())
        return root

    def derivative(self) -> FunctionNode:
        root = SubNode()
        root.pack(self.link[0].derivative(), 0)
        root.pack(self.link[1].derivative(), 1)
        return root

    def expression(self, symbol):
        return self.link[0].expression(symbol) - self.link[1].expression(symbol)


class MulNode(FunctionNode):
    def __init__(self, lnksize=0, resizable=True):
        super(MulNode, self).__init__(lnksize, resizable)

    def calculate(self, arg: Variable) -> Variable:
        ret = Variable(1)
        for lnk in self.link:
            ret *= lnk.calculate(arg)
        return ret

    def cpy(self) -> FunctionNode:
        root = MulNode()
        for lnk in self.link:
            if lnk is not None:
                root.append(lnk.cpy())
        return root

    def derivative(self) -> FunctionNode:
        root = AddNode()

        for i in range(self._lnksize):
            mul = MulNode()
            for j in range(self._lnksize):
                if i == j:
                    mul.append(self.link[j].derivative())
                else:
                    mul.append(self.link[j].cpy())
            root.append(mul)

        return root

    def expression(self, symbol):
        ret = 1
        for lnk in self.link:
            ret *= lnk.expression(symbol)
        return ret


class DivNode(FunctionNode):
    def __init__(self):
        super(DivNode, self).__init__(lnksize=2)

    def calculate(self, arg: Variable) -> Variable:
        cache = self.link[1].calculate(arg)
        if cache.get() == 0:
            return Variable(NAN)
        return self.link[0].calculate(arg) / cache

    def cpy(self) -> FunctionNode:
        root = DivNode()
        for lnk in self.link:
            if lnk is not None:
                root.append(lnk.cpy())
        return root

    def derivative(self) -> FunctionNode:
        root = DivNode()
        sub1 = SubNode()
        mul1 = MulNode()
        mul2 = MulNode()
        mul3 = MulNode()

        root.pack(sub1, 0)
        root.pack(mul3, 1)

        sub1.pack(mul1, 0)
        sub1.pack(mul2, 1)

        mul1.append(self.link[0].derivative())
        mul1.append(self.link[1].cpy())

        mul2.append(self.link[0].cpy())
        mul2.append(self.link[1].derivative())

        mul3.append(self.link[1].cpy())
        mul3.append(self.link[1].cpy())

        return root

    def expression(self, symbol):
        return self.link[0].expression(symbol) / self.link[1].expression(symbol)


class NegNode(FunctionNode):
    def __init__(self):
        super(NegNode, self).__init__(lnksize=1)

    def calculate(self, arg: Variable) -> Variable:
        return self.link[0].calculate(arg) * (-1)

    def cpy(self) -> FunctionNode:
        root = NegNode()
        for lnk in self.link:
            if lnk is not None:
                root.append(lnk.cpy())
        return root

    def derivative(self) -> FunctionNode:
        root = NegNode()
        root.pack(self.link[0].derivative(), 0)
        return root

    def expression(self, symbol):
        return -1 * self.link[0].expression(symbol)


class LinearNode(FunctionNode):
    def __init__(self, value: (int, float)):
        super(LinearNode, self).__init__(lnksize=1)
        self.value = float(value)

    def calculate(self, arg: Variable) -> Variable:
        return self.link[0].calculate(arg) * self.value

    def cpy(self) -> FunctionNode:
        root = LinearNode(self.value)
        for lnk in self.link:
            if lnk is not None:
                root.append(lnk.cpy())
        return root

    def derivative(self) -> FunctionNode:
        root = LinearNode(self.value)
        root.pack(self.link[0].derivative(), 0)
        return root

    def expression(self, symbol):
        return self.value * self.link[0].expression(symbol)


class ExpNode(FunctionNode):
    def __init__(self, value=E):
        super(ExpNode, self).__init__(lnksize=1)
        if isinstance(value, (int, float)):
            self.value = float(value)
        else:
            raise TypeError("'value' needs to be int or float")

    def calculate(self, arg: Variable) -> Variable:
        return Variable(self.value ** self.link[0].calculate(arg).get())

    def cpy(self) -> FunctionNode:
        root = ExpNode(self.value)
        for lnk in self.link:
            if lnk is not None:
                root.append(lnk.cpy())
        return root

    def derivative(self) -> FunctionNode:
        root = MulNode()

        root.append(ScalarNode(math.log(self.value, E)))
        root.append(self.link[0].derivative())
        root.append(self.cpy())

        return root

    def expression(self, symbol):
        return self.value ** self.link[0].expression(symbol)


class SqrNode(FunctionNode):
    def __init__(self, value=2.0):
        super(SqrNode, self).__init__(lnksize=1)
        if isinstance(value, (int, float)):
            self.value = float(value)
        else:
            raise TypeError("'value' needs to be int or float")

    def calculate(self, arg: Variable) -> Variable:
        if self.value < 1 and arg.get() < 0:
            return Variable(NAN)
        return Variable(self.link[0].calculate(arg).get() ** self.value)

    def cpy(self) -> FunctionNode:
        root = SqrNode(self.value)
        for lnk in self.link:
            if lnk is not None:
                root.append(lnk.cpy())
        return root

    def derivative(self) -> FunctionNode:
        root = MulNode()
        scl = ScalarNode(value=self.value)
        sqrnd = SqrNode(value=self.value-1)

        root.append(scl)
        root.append(self.link[0].derivative())
        root.append(sqrnd)

        sqrnd.pack(self.link[0].cpy())

        return root

    def expression(self, symbol):
        return self.link[0].expression(symbol) ** self.value


class LogNode(FunctionNode):
    def __init__(self, value=E):
        super(LogNode, self).__init__(lnksize=1)

        if isinstance(value, (int, float)):
            if value <= 0:
                raise TypeError("'value' of 'LogNode' cannot be negative number or 0")
            self.value = float(value)
        else:
            raise TypeError("'value' of 'LogNode' needs to be int or float")

    def calculate(self, arg: Variable) -> Variable:
        cache = self.link[0].calculate(arg).get()
        if math.isnan(cache) or cache <= 0:
            return Variable(NAN)
        return Variable(cache).log(base=self.value)

    def cpy(self) -> FunctionNode:
        root = LogNode(self.value)
        for lnk in self.link:
            if lnk is not None:
                root.append(lnk.cpy())
        return root

    def derivative(self) -> FunctionNode:
        if self.value == E:
            root = DivNode()
            root.pack(self.link[0].derivative(), 0)
            root.pack(self.link[0].cpy(), 1)
        else:
            root = LinearNode(value=(1/math.log(self.value, E)))
            div = DivNode()

            root.pack(div, 0)

            div.pack(self.link[0].derivative(), 0)
            div.pack(self.link[0].cpy(), 1)

        return root

    def expression(self, symbol):
        return sp.log(self.link[0].expression(symbol), self.value)


class SinNode(FunctionNode):
    def __init__(self):
        super(SinNode, self).__init__(lnksize=1)

    def calculate(self, arg: Variable) -> Variable:
        return self.link[0].calculate(arg).sin()

    def cpy(self) -> FunctionNode:
        root = SinNode()
        for lnk in self.link:
            if lnk is not None:
                root.append(lnk.cpy())
        return root

    def derivative(self) -> FunctionNode:
        root = MulNode()
        cosnd = CosNode()

        root.append(self.link[0].derivative())
        root.append(cosnd)

        cosnd.pack(self.link[0].cpy())

        return root

    def expression(self, symbol):
        return sp.sin(self.link[0].expression(symbol))


class CosNode(FunctionNode):
    def __init__(self):
        super(CosNode, self).__init__(lnksize=1)

    def calculate(self, arg: Variable) -> Variable:
        return self.link[0].calculate(arg).cos()

    def cpy(self) -> FunctionNode:
        root = CosNode()
        for lnk in self.link:
            if lnk is not None:
                root.append(lnk.cpy())
        return root

    def derivative(self) -> FunctionNode:
        root = MulNode()
        linnd = LinearNode(value=-1)
        sinnd = SinNode()

        root.append(linnd)
        root.append(sinnd)

        linnd.pack(self.link[0].derivative(), 0)
        sinnd.pack(self.link[0].cpy(), 0)

        return root

    def expression(self, symbol):
        return sp.cos(self.link[0].expression(symbol))


class HeavisideNode(FunctionNode):
    def __init__(self, eps=STD_EPS):
        super(HeavisideNode, self).__init__(lnksize=1)
        self.eps = eps

    def calculate(self, arg: Variable) -> Variable:
        cache = self.link[0].calculate(arg)
        if cache.get() >= self.eps:
            return Variable(1)
        elif cache.get() <= -1 * self.eps:
            return Variable(0)
        else:
            return Variable(0.5 * (cache.get() / self.eps + 1))

    def cpy(self) -> FunctionNode:
        root = HeavisideNode(eps=self.eps)
        for lnk in self.link:
            if lnk is not None:
                root.append(lnk.cpy())
        return root

    def derivative(self) -> FunctionNode:
        root = MulNode()
        root.append(DiracDeltaNode(eps=self.eps).append(self.link[0].cpy()))
        root.append(self.link[0].derivative())
        return root

    def expression(self, symbol):
        return sp.Heaviside(self.link[0].expression(symbol))


class DiracDeltaNode(FunctionNode):
    def __init__(self, eps=STD_EPS):
        super(DiracDeltaNode, self).__init__(lnksize=1)
        self.eps = eps

    def calculate(self, arg: Variable) -> Variable:
        cache = self.link[0].calculate(arg)
        if -1 * self.eps <= cache.get() <= self.eps:
            return Variable(1 / (2 * self.eps))
        else:
            return Variable(0)

    def cpy(self) -> FunctionNode:
        root = DiracDeltaNode(eps=self.eps)
        for lnk in self.link:
            if lnk is not None:
                root.append(lnk.cpy())
        return root

    def derivative(self) -> FunctionNode:
        raise TypeError("'DiracDeltaNode' does not have derivative function")

    def expression(self, symbol):
        return sp.DiracDelta(self.link[0].expression(symbol))


class Function(object):
    def __init__(self, root, eps=STD_EPS):
        if isinstance(root, FunctionNode):
            self.root = root.cpy()
        elif isinstance(root, (int, float)):
            self.root = ScalarNode(root)
        else:
            raise TypeError("'root' needs to be int, float, or 'FunctionNode'")

        if isinstance(eps, (int, float)):
            self._eps = float(eps)
            self.root.set_eps(self._eps)
        else:
            raise TypeError("'eps' needs to be int or float")

    def set_eps(self, eps=STD_EPS):
        self._eps = eps
        self.root.set_eps(self._eps)

    def calculate(self, arg: Variable) -> Variable:
        return self.root.calculate(arg)

    def cpy(self):
        return Function(root=self.root.cpy())

    def derivative(self):
        return Function(root=self.root.derivative())

    def expression(self, symbol):
        return self.root.expression(symbol)

    def plot(self, xrange=10, yrange=5, yscale=5/4, neg=True, grid=True):
        if not isinstance(xrange, (int, float)):
            raise TypeError("'xrange' needs to be int or float")

        if not (isinstance(yrange, (int, float)) or yrange == MAX):
            raise TypeError("'yrange' needs to be int or float")

        if not isinstance(yscale, (int, float)):
            raise TypeError("'yscale' needs to be int or float")

        if not isinstance(neg, bool):
            raise TypeError("'neg' needs to be bool")

        x = np.linspace(0 if not neg else -xrange, xrange, math.ceil(50 / self._eps))
        y = [self.calculate(Variable(val)).get() for val in x]

        xlim = xrange
        ylim = yrange
        if ylim == 'MAX':
            y_min, y_max = min(y), max(y)
            ylim = max((y_min-1)*yscale, (y_max+1)*yscale)

        plt.plot(x, y, label='signal')
        plt.grid(grid)
        axes = plt.gca()
        axes.set_xlim([0 if not neg else -xlim, xlim])
        axes.set_ylim([-ylim, ylim])
        plt.xlabel('input')
        plt.ylabel('value')
        plt.legend()
        plt.show()

    def __add__(self, other):
        ret = Function(root=AddNode())
        if isinstance(other, Function):
            ret.root.append(self.root.cpy()).append(other.root.cpy())
        if isinstance(other, (int, float)):
            ret.root.append(self.root.cpy()).append(ScalarNode(other))
        if isinstance(other, Variable):
            ret.root.append(self.root.cpy()).append(ScalarNode(other.get()))
        return ret

    def __sub__(self, other):
        ret = Function(root=SubNode())
        if isinstance(other, Function):
            ret.root.append(self.root.cpy()).append(other.root.cpy())
        if isinstance(other, (int, float)):
            ret.root.append(self.root.cpy()).append(ScalarNode(other))
        if isinstance(other, Variable):
            ret.root.append(self.root.cpy()).append(ScalarNode(other.get()))
        return ret

    def __mul__(self, other):
        ret = Function(root=MulNode())
        if isinstance(other, Function):
            ret.root.append(self.root.cpy()).append(other.root.cpy())
        if isinstance(other, (int, float)):
            ret.root.append(self.root.cpy()).append(ScalarNode(other))
        if isinstance(other, Variable):
            ret.root.append(self.root.cpy()).append(ScalarNode(other.get()))
        return ret

    def __div__(self, other):
        ret = Function(root=DivNode())
        if isinstance(other, Function):
            ret.root.append(self.root.cpy()).append(other.root.cpy())
        if isinstance(other, (int, float)):
            ret.root.append(self.root.cpy()).append(ScalarNode(other))
        if isinstance(other, Variable):
            ret.root.append(self.root.cpy()).append(ScalarNode(other.get()))
        return ret

    def __radd__(self, other):
        return self.__add__(other)

    def __rsub__(self, other):
        ret = Function(root=SubNode())
        if isinstance(other, Function):
            ret.root.append(other.root.cpy()).append(self.root.cpy())
        if isinstance(other, (int, float)):
            ret.root.append(ScalarNode(other)).append(self.root.cpy())
        if isinstance(other, Variable):
            ret.root.append(ScalarNode(other.get())).append(self.root.cpy())
        return ret

    def __rmul__(self, other):
        return self.__mul__(other)

    def __rdiv__(self, other):
        ret = Function(root=DivNode())
        if isinstance(other, Function):
            ret.root.append(other.root.cpy()).append(self.root.cpy())
        if isinstance(other, (int, float)):
            ret.root.append(ScalarNode(other)).append(self.root.cpy())
        if isinstance(other, Variable):
            ret.root.append(ScalarNode(other.get())).append(self.root.cpy())
        return ret

    def __truediv__(self, other):
        return self.__div__(other)

    def __floordiv__(self, other):
        return self.__div__(other)

    def __rtruediv__(self, other):
        return self.__rdiv__(other)

    def __rfloordiv__(self, other):
        return self.__rdiv__(other)

    def __neg__(self):
        ret = Function(root=NegNode().append(self.root.cpy()))
        return ret

    def __pow__(self, power, modulo=None):
        ret = Function(root=SqrNode())
        if isinstance(power, Function):
            if not isinstance(power.root, ScalarNode):
                raise TypeError("operator '**' does not support operation between 'Function' / 'Function'")
            ret.root.value = power.root.value
            ret.root.append(self.root.cpy())
        if isinstance(power, (int, float)):
            ret.root.value = power
            ret.root.append(self.root.cpy())
        if isinstance(power, Variable):
            ret.root.value = power.get()
            ret.root.append(self.root.cpy())
        return ret

    def __rpow__(self, power, modulo=None):
        ret = Function(root=ExpNode())
        if isinstance(power, Function):
            if not isinstance(power.root, ScalarNode):
                raise TypeError("operator '**' does not support operation between 'Function' / 'Function'")
            ret.root.value = power.root.value
            ret.root.append(self.root.cpy())
        if isinstance(power, (int, float)):
            ret.root.value = power
            ret.root.append(self.root.cpy())
        if isinstance(power, Variable):
            ret.root.value = power.get()
            ret.root.append(self.root.cpy())
        return ret


class ParametricEquation(object):
    def __init__(self, *roots):
        self.roots = list(roots)
        self.dim = len(self.roots)

    def append(self, root: FunctionNode):
        self.roots.append(root.cpy())
        self.dim = len(self.roots)
        return self

    def calculate(self, arg: Variable) -> tuple:
        return tuple([root.calculate(arg) for root in self.roots])

    def cpy(self):
        ret = ParametricEquation()
        for root in self.roots:
            ret.append(root.cpy())
        return ret

    def derivative(self):
        ret = ParametricEquation()
        for root in self.roots:
            ret.append(root.derivative())
        return ret

    def __add__(self, other):
        if isinstance(other, ParametricEquation) and other.dim == self.dim:
            roots = [AddNode().append(r1.cpy()).append(r2.cpy()) for r1, r2 in zip(self.roots, other.roots)]
            ret = ParametricEquation(*roots)
            return ret
        else:
            raise TypeError("'dim' needs to be same")

    def __sub__(self, other):
        if isinstance(other, ParametricEquation) and other.dim == self.dim:
            roots = [SubNode().append(r1.cpy()).append(r2.cpy()) for r1, r2 in zip(self.roots, other.roots)]
            ret = ParametricEquation(*roots)
            return ret
        else:
            raise TypeError("'dim' needs to be same")

    def __mul__(self, other):
        if isinstance(other, ParametricEquation) and other.dim == self.dim:
            roots = [MulNode().append(r1.cpy()).append(r2.cpy()) for r1, r2 in zip(self.roots, other.roots)]
            ret = ParametricEquation(*roots)
            return ret
        else:
            raise TypeError("'dim' needs to be same")

    def __div__(self, other):
        if isinstance(other, ParametricEquation) and other.dim == self.dim:
            roots = [DivNode().append(r1.cpy()).append(r2.cpy()) for r1, r2 in zip(self.roots, other.roots)]
            ret = ParametricEquation(*roots)
            return ret
        else:
            raise TypeError("'dim' needs to be same")

    def __truediv__(self, other):
        if isinstance(other, ParametricEquation) and other.dim == self.dim:
            roots = [DivNode().append(r1.cpy()).append(r2.cpy()) for r1, r2 in zip(self.roots, other.roots)]
            ret = ParametricEquation(*roots)
            return ret
        else:
            raise TypeError("'dim' needs to be same")

    def __floordiv__(self, other):
        if isinstance(other, ParametricEquation) and other.dim == self.dim:
            roots = [DivNode().append(r1.cpy()).append(r2.cpy()) for r1, r2 in zip(self.roots, other.roots)]
            ret = ParametricEquation(*roots)
            return ret
        else:
            raise TypeError("'dim' needs to be same")


# Function redefined with general form
# These functions returns 'FunctionNode' and can be used for the root of 'Function'

def sqr(nd=VarNode(), value=2):
    if isinstance(nd, FunctionNode):
        return SqrNode(value=value).pack(nd.cpy())
    if isinstance(nd, (int, float)):
        return ScalarNode(value=nd ** value)
    raise TypeError("invalid 'nd'")


def exp(nd=VarNode(), value=E):
    if isinstance(nd, FunctionNode):
        return ExpNode(value=value).pack(nd.cpy())
    if isinstance(nd, (int, float)):
        return ScalarNode(value=value ** nd)
    raise TypeError("invalid 'nd'")


def sin(nd=VarNode()):
    if isinstance(nd, FunctionNode):
        return SinNode().pack(nd.cpy())
    if isinstance(nd, (int, float)):
        return ScalarNode(value=math.sin(nd))
    raise TypeError("invalid 'nd'")


def cos(nd=VarNode()):
    if isinstance(nd, FunctionNode):
        return CosNode().pack(nd.cpy())
    if isinstance(nd, (int, float)):
        return ScalarNode(value=math.cos(nd))
    raise TypeError("invalid 'nd'")


def heaviside(nd=VarNode(), eps=STD_EPS):
    return HeavisideNode(eps=eps).append(nd.cpy())


def dirac_delta(nd=VarNode(), eps=STD_EPS):
    return DiracDeltaNode(eps=eps).append(nd.cpy())


if __name__ == '__main__':
    # FunctionNode examples

    """root = SubNode()  # -
    root.pack(SqrNode(value=3).pack(VarNode()), 0)  # x^3
    root.pack(LinearNode(value=2).pack(VarNode()), 1)  # 2x

    func = Function(root=root)  # x^3 - 2x
    derv = func.derivative()  # 3x^2 - 2

    x = Variable(2)
    y = func.calculate(x)
    dy = derv.calculate(x)

    print(x.get(), y.get(), dy.get())

    func.plot(xrange=10, yrange=MAX)
    derv.plot(xrange=10, yrange=MAX)"""

    # ComplexVar examples

    """a = ComplexVar(1, 2)
    b = ComplexVar().set(real=3, imag=1)

    print(a)
    print(b)

    print(a + b)
    print(a - b)
    print(a * b)
    print(a / b)"""

    # HeavisideNode examples

    """root = HeavisideNode()
    func = Function(root=root)
    func.plot(xrange=10)
    func.derivative().plot(xrange=10)"""

    # Sinusoidal signal with Function

    """root = MulNode().append(CosNode().append(VarNode())).append(HeavisideNode())
    func = Function(root=root)
    func.plot(xrange=10, yrange=5, neg=True)
    func.derivative().plot(xrange=10, yrange=10, neg=True)"""

    # Function defined with general operators

    """var = VarNode()
    func = Function(var * E ** var)
    func.plot(xrange=10, yrange=MAX, neg=True)
    func.derivative().plot(xrange=10, yrange=MAX, neg=True)"""

    """func = Function(var ** 3 - 7 * var ** 2)
    func.plot(xrange=10, yrange=MAX, neg=True)"""

    """func = Function(var * heaviside())
    func.plot(xrange=10, yrange=MAX, neg=True)
    func.derivative().plot(xrange=10, yrange=5, neg=True)"""
