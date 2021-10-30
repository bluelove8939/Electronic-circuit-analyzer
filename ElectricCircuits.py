import os
from typing import List, Dict, Tuple
import sympy as sp
import UnivariateFunction as Uf


current_dir = os.path.abspath(os.path.split(__file__)[0])

DEFAULT = 'DEFAULT'
t_var, s_var = sp.symbols('t, s')


class BaseElement(object):
    def __init__(self, portsize=2):
        self.portsize = portsize

    def cpy(self):
        return BaseElement()

    def get_attr(self, target: str, port_id1: int, port_id2: int, **conditions):
        pass


class Resistor(BaseElement):
    def __init__(self, resistance: (int, float)):
        self.resistance = resistance
        super(Resistor, self).__init__(portsize=2)

    def cpy(self) -> BaseElement:
        return Resistor(self.resistance)

    def get_attr(self, target: str, port_id1: int, port_id2: int, **conditions):
        if target in conditions.keys():
            raise TypeError("Invalid operation: condition already includes target")
        if port_id1 < 0 or port_id2 >= self.portsize:
            raise TypeError(f"Invalid operation: invalid port_id1 {port_id1}")
        if port_id2 < 0 or port_id2 >= self.portsize:
            raise TypeError(f"Invalid operation: invalid port_id2 {port_id2}")
        if port_id1 == port_id2:
            raise TypeError("Invalid operation: cannot define attribute with same port_id")

        if target == 'current':
            if 'voltage' not in conditions.keys():
                return None

            if isinstance(conditions['voltage'], tuple) and len(conditions['voltage']) == 3:
                vol_port1, vol_port2 = conditions['voltage'][1:]
                if vol_port1 < 0 or vol_port1 >= self.portsize or \
                        vol_port1 < 0 or vol_port1 >= self.portsize:
                    raise TypeError(f"Invalid operation: voltage cannot be defined between ({vol_port1}, {vol_port2})")
                else:
                    coef = 1 if (port_id1, port_id2) == (vol_port1, vol_port2) else -1
                if isinstance(conditions['voltage'][0], Uf.Function):
                    vol = conditions['voltage'][0].expression(t_var)
                else:
                    vol = conditions['voltage'][0]
                return coef * vol / self.resistance

            return None

        if target == 'voltage':
            if 'current' not in conditions.keys():
                return None

            if isinstance(conditions['current'], tuple) and len(conditions['current']) == 2:
                cur_port = conditions['current'][1]
                if cur_port < 0 or cur_port >= self.portsize:
                    raise TypeError(f"Invalid operation: current cannot be defined at port {cur_port}")
                coef = 1 if cur_port == port_id2 else -1
                if isinstance(conditions['current'][0], Uf.Function):
                    cur = conditions['current'][0].expression(t_var)
                else:
                    cur = conditions['current'][0]
                return coef * self.resistance * cur

            return None

        if target == 'resistance':
            return self.resistance

        return None


def signal_dc(state: (int, float)) -> Uf.Function:
    return Uf.Function(Uf.heaviside(Uf.VarNode()) * state)


def signal_ac(amplitude: (int, float), ang_freq: (int, float), phase: (int, float) = 0.0) -> Uf.Function:
    return Uf.Function(amplitude * Uf.cos(ang_freq * Uf.VarNode() + phase))


class VoltageSource(BaseElement):
    def __init__(self, voltage: Uf.Function):
        self.voltage = voltage
        super(VoltageSource, self).__init__(portsize=2)

    def cpy(self) -> BaseElement:
        return VoltageSource(self.voltage)

    def get_attr(self, target: str, port_id1: int, port_id2: int, **conditions):
        if target in conditions.keys():
            raise TypeError("Invalid operation: condition already includes target")
        if port_id1 < 0 or port_id2 >= self.portsize:
            raise TypeError(f"Invalid operation: invalid port_id1 {port_id1}")
        if port_id2 < 0 or port_id2 >= self.portsize:
            raise TypeError(f"Invalid operation: invalid port_id2 {port_id2}")
        if port_id1 == port_id2:
            raise TypeError("Invalid operation: cannot define attribute with same port_id")

        if target == 'current':
            return None
        if target == 'voltage':
            if port_id1 == 0:
                return self.voltage.expression(t_var)
            else:
                return -1 * self.voltage.expression(t_var)

        return None


class Analyzer(object):
    class Node(object):
        def __init__(self, name: str, lnksize=0):
            self.name = name
            self.lnk: List[Tuple[str, int] or None] = [None for _ in range(lnksize)]
            self.searching_finished: bool = False
            self.visited = None
            self.path_voltage_cache = False
            self.current = []
            self.current_symbol = []

        def port(self, port_id: int):
            return self.name, port_id

        def makelink(self, node, port_id):
            names = [port_info[0] for port_info in self.lnk]
            if node.name in names:
                raise TypeError(f"invalid operation: link is already made with the node '{node.name}'")
            if isinstance(node, Analyzer.ElementNode):
                self.lnk.append((node.name, port_id))
                node.lnk[port_id] = (self.name, self.lnk.index((node.name, port_id)))
            else:
                raise TypeError("invalid operation: 'node' needs to be 'Analyzer.ElementNode'")

        def init_visited(self):
            self.searching_finished = False
            self.visited = None
            self.path_voltage_cache = False

        def init_current(self):
            self.current_symbol = [sp.symbols(f"i_{self.lnk[port][0]}_{self.lnk[port][1]}", cls=sp.Function)
                                   for port in range(len(self.lnk))]
            self.current = [-1 * self.current_symbol[port_id](t_var) for port_id in range(len(self.lnk))]

        def __str__(self):
            return f"node {self.name}: {' '.join([str(port_info) for port_info in self.lnk])}"

    class ElementNode(Node):
        def __init__(self, name: str, body: BaseElement):
            super(Analyzer.ElementNode, self).__init__(name, lnksize=body.portsize)
            self.body = body

        def init_current(self):
            self.current_symbol = [sp.symbols(f"i_{self.name}_{port}", cls=sp.Function)
                                   for port in range(len(self.lnk))]
            self.current = [self.current_symbol[port_id](t_var) for port_id in range(len(self.lnk))]

        def __str__(self):
            return f"element {self.name}: {' '.join([str(port_info) for port_info in self.lnk])}"

    def __init__(self):
        self.elements: Dict[str, Analyzer.ElementNode] = dict()
        self.nodelist: Dict[str, Analyzer.Node] = dict()
        self.equations = []
        self.solution = dict()

    def add(self, body: BaseElement, name: str = 'default') -> ElementNode:
        if name == 'default':
            name = self.create_new_element_name()
        else:
            if name in self.elements.keys():
                raise TypeError(f"name '{name}' already defined")

        self.elements[name] = Analyzer.ElementNode(name, body)

        return self.elements[name]

    def create_new_node_name(self):
        name, num = 'node', 1
        while name + str(num) in self.nodelist.keys():
            num += 1
        return name + str(num)

    def create_new_element_name(self):
        name, num = 'element', 1
        while name + str(num) in self.nodelist.keys():
            num += 1
        return name + str(num)

    def link(self, *port: Tuple[str, int]):
        if len(port) < 2:
            raise TypeError("at least 2 ports are needed")

        names = [port_info[0] for port_info in port]
        for idx, name in enumerate(names):
            if name in names[:idx]:
                raise TypeError("'same element cannot be linked")

        node = None
        for name, port_id in port:
            if node is None:
                if self.elements[name].lnk[port_id] is None:
                    node = Analyzer.Node(name=self.create_new_node_name())
                    self.nodelist[node.name] = node
                    node.makelink(self.elements[name], port_id)
                else:
                    node = self.nodelist[name].lnk[port_id]
            else:
                if self.elements[name].lnk[port_id] is None:
                    node.makelink(self.elements[name], port_id)
                else:
                    new_node = Analyzer.Node(name=self.create_new_node_name())
                    old_node = self.nodelist[self.elements[name].lnk[port_id][0]]

                    for new_name, new_port_id in node.lnk + self.nodelist[self.elements[name].lnk[port_id][0]].lnk:
                        new_node.makelink(self.elements[new_name], new_port_id)

                    self.nodelist.pop(old_node.name)
                    self.nodelist.pop(node.name)
                    node = new_node
                    self.nodelist[node.name] = node

    def analyze(self):
        if len(self.nodelist) < 2:
            raise TypeError(f"Invalid operation: at least 2 elements required ({len(self.nodelist)}")

        for node in self.nodelist.values():
            node.init_current()

        for element in self.elements.values():
            element.init_current()

        self.equations = []
        print('start searching nodes...')
        self.search_node(node=self.nodelist[list(self.nodelist.keys())[0]])
        print('finished searching nodes')
        print(f'start calaulating {len(self.equations)} equations...')
        self.solution = sp.solve(self.equations)[0]
        print('solution generated')

    def search_node(self, node: Node, port: int = -1, path_voltage=None):
        if node.searching_finished:
            return

        if port == -1:
            path_voltage = 0
        elif port < 0 or port >= len(node.lnk):
            raise TypeError(f"Invalid operation: port {port} not found")
        else:
            if node.visited is not None:  # KVL (if a loop is detected)
                if path_voltage is not None:
                    self.equations.append(sp.Eq(path_voltage - node.path_voltage_cache, 0))
                return
            else:
                node.visited = True
                node.path_voltage_cache = path_voltage

        for lnk_port_id, element_info in enumerate(node.lnk):
            if element_info is None:
                continue
            if lnk_port_id != port:
                self.search_element(self.elements[element_info[0]], element_info[1],
                                    path_voltage=path_voltage)

        # KCL (with any other nodes)
        expr = 0
        for cur in node.current:
            expr += cur
        self.equations.append(sp.Eq(expr, 0))

        node.visited = None
        node.searching_finished = True

    def search_element(self, element: ElementNode, port: int = -1, path_voltage=None):
        if element.searching_finished:
            return

        if port == -1:
            path_voltage = 0
        elif port < 0 or port >= len(element.lnk):
            raise TypeError(f"Invalid operation: port {port} not found")
        else:
            if element.visited is not None:  # KVL (if a loop is detected)
                if path_voltage is not None:
                    self.equations.append(sp.Eq(path_voltage - element.path_voltage_cache, 0))
                return
            else:
                element.visited = True
                element.path_voltage_cache = path_voltage

        for lnk_port_id, node_info in enumerate(element.lnk):
            if node_info is None:
                continue
            if lnk_port_id != port:
                edge_voltage = element.body.get_attr('voltage', port, lnk_port_id,
                                                     current=(element.current[port], port))
                new_path_voltage = path_voltage + edge_voltage if edge_voltage is not None else None
                self.search_node(self.nodelist[node_info[0]], node_info[1],
                                 path_voltage=new_path_voltage)

        # KCL (with any other nodes)
        expr = 0
        for cur in element.current:
            expr += cur
        self.equations.append(sp.Eq(expr, 0))

        element.visited = None
        element.searching_finished = True
