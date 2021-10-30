from ElectricCircuits import *
from UnivariateFunction import *
import sympy as sp

if __name__ == '__main__':
    analyzer = Analyzer()

    V1 = analyzer.add(VoltageSource(voltage=signal_dc(5)), 'V1')
    R1 = analyzer.add(Resistor(5), 'R1')
    R2 = analyzer.add(Resistor(5), 'R2')
    R3 = analyzer.add(Resistor(5), 'R3')
    R4 = analyzer.add(Resistor(5), 'R4')
    R5 = analyzer.add(Resistor(5), 'R5')
    R6 = analyzer.add(Resistor(5), 'R6')
    R7 = analyzer.add(Resistor(5), 'R7')
    R8 = analyzer.add(Resistor(5), 'R8')
    R9 = analyzer.add(Resistor(5), 'R9')
    R10 = analyzer.add(Resistor(5), 'R10')

    analyzer.link(R1.port(0), V1.port(0))
    analyzer.link(R1.port(1), R2.port(0), R3.port(0), R5.port(0))
    analyzer.link(R3.port(1), R4.port(0))
    analyzer.link(R2.port(1), R4.port(1), R5.port(1), R6.port(0), R7.port(0))
    analyzer.link(R6.port(1), R8.port(0), R9.port(0))
    analyzer.link(R7.port(1), R8.port(1), R10.port(0))
    analyzer.link(R9.port(1), R10.port(1), V1.port(1))

    """V1 = analyzer.add(VoltageSource(voltage=signal_dc(10)), 'V1')
    R1 = analyzer.add(Resistor(5), 'R1')
    R2 = analyzer.add(Resistor(5), 'R2')

    analyzer.link(R1.port(0), V1.port(0))
    analyzer.link(R1.port(1), V1.port(1))
    analyzer.link(R2.port(0), R1.port(1))"""

    """V1 = analyzer.add(VoltageSource(voltage=signal_ac(10, PI, 0)), 'V1')
    R1 = analyzer.add(Resistor(5), 'R1')

    analyzer.link(R1.port(0), V1.port(0))
    analyzer.link(R1.port(1), V1.port(1))"""

    for name in analyzer.elements.keys():
        print(analyzer.elements[name])

    for name in analyzer.nodelist.keys():
        print(analyzer.nodelist[name])

    analyzer.analyze()
    print(analyzer.solution[R1.current_symbol[1](t_var)])

    """var = VarNode()

    func = Function(cos(var))
    expr = func.expression(t_var)
    print(sp.laplace_transform(expr, t_var, s_var, noconds=True))  # s/(s**2 + 1)

    func = Function(1 / var)
    expr = func.expression(s_var)
    print(sp.inverse_laplace_transform(expr, s_var, t_var, noconds=True))  # 1.0*Heaviside(t)"""

    """vc = VoltageSource(Function(5))
    print(vc.get_attr('voltage', 0, 1).expression(t_var))
    print(vc.get_attr('voltage', 1, 0).expression(t_var))

    rs = Resistor(5)
    print(rs.get_attr('voltage', 0, 1, current=Function(5)).expression(t_var))
    print(rs.get_attr('current', 0, 1, voltage=Function(5)).expression(t_var))"""
