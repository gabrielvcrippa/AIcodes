import math
import sys
import copy
import heapq
import time
from collections import deque


# |=====================|
# | CÓDIGO DO PROFESSOR |
# |=====================|

class Problem(object):
    def __init__(self, initial=None, goal=None, **kwds):
        self.__dict__.update(initial=initial, goal=goal, **kwds)

    def actions(self, state):        raise NotImplementedError
    def result(self, state, action): raise NotImplementedError
    def is_goal(self, state):        return state == self.goal
    def action_cost(self, s, a, s1): return 1
    def h(self, node):               return 0


class Node:
    def __init__(self, state, parent=None, action=None, path_cost=0):
        self.__dict__.update(state=state, parent=parent, action=action, path_cost=path_cost)

    def __repr__(self): return '<{}>'.format(self.state)
    def __len__(self):  return 0 if self.parent is None else (1 + len(self.parent))
    def __lt__(self, other): return self.path_cost < other.path_cost


failure = Node('failure', path_cost=math.inf)
cutoff  = Node('cutoff',  path_cost=math.inf)


def expand(problem, node):
    s = node.state
    for action in problem.actions(s):
        s1 = problem.result(s, action)
        if s1 is not None:
            cost = node.path_cost + problem.action_cost(s, action, s1)
            yield Node(s1, node, action, cost)


def path_actions(node):
    if node.parent is None:
        return []
    return path_actions(node.parent) + [node.action]


def path_states(node):
    if node in (cutoff, failure, None):
        return []
    return path_states(node.parent) + [node.state]


def is_cycle(node, k=30):
    def find_cycle(ancestor, k):
        return (ancestor is not None and k > 0 and
                (ancestor.state == node.state or find_cycle(ancestor.parent, k - 1)))
    return find_cycle(node.parent, k)


FIFOQueue = deque
LIFOQueue = list

class PriorityQueue:
    """A queue in which the item with minimum f(item) is always popped first."""

    def __init__(self, items=(), key=lambda x: x):
        self.key = key
        self.items = []
        for item in items:
            self.add(item)

    def add(self, item):
        pair = (self.key(item), item)
        heapq.heappush(self.items, pair)

    def pop(self):
        return heapq.heappop(self.items)[1]

    def top(self): return self.items[0][1]

    def __len__(self): return len(self.items)


def breadth_first_search(problem):
    nos_explorados = 0
    node = Node(problem.initial)
    if problem.is_goal(problem.initial):
        return node, nos_explorados
    frontier = FIFOQueue([node])
    reached = {problem.initial}
    while frontier:
        node = frontier.pop()
        nos_explorados += 1
        for child in expand(problem, node):
            s = child.state
            if problem.is_goal(s):
                return child, nos_explorados
            if s not in reached:
                reached.add(s)
                frontier.appendleft(child)
    return failure, nos_explorados


def depth_limited_search(problem, limit=10):
    nos_explorados = 0
    frontier = LIFOQueue([Node(problem.initial)])
    result = failure
    while frontier:
        node = frontier.pop()
        nos_explorados += 1
        if problem.is_goal(node.state):
            return node, nos_explorados
        elif len(node) >= limit:
            result = cutoff
        elif not is_cycle(node):
            for child in expand(problem, node):
                frontier.append(child)
    return result, nos_explorados


def iterative_deepening_search(problem):
    total_nos = 0
    for limit in range(1, sys.maxsize):
        result, nos = depth_limited_search(problem, limit)
        total_nos += nos
        if result != cutoff:
            return result, total_nos


def depth_first_recursive_search(problem, node=None, nos_explorados=None):
    if nos_explorados is None:
        nos_explorados = [0]
    if node is None:
        node = Node(problem.initial)
    if problem.is_goal(node.state):
        return node, nos_explorados[0]
    elif is_cycle(node):
        return failure, nos_explorados[0]
    else:
        for child in expand(problem, node):
            nos_explorados[0] += 1
            result, nos = depth_first_recursive_search(problem, child, nos_explorados)
            if result != failure:
                return result, nos
        return failure, nos_explorados[0]
    

def best_first_search(problem, f):
    nos_explorados = 0
    node = Node(problem.initial)
    frontier = PriorityQueue([node], key=f)
    reached = {problem.initial: node}
    while frontier:
        node = frontier.pop()
        nos_explorados += 1
        if problem.is_goal(node.state):
            return node, nos_explorados
        for child in expand(problem, node):
            s = child.state
            if s not in reached or child.path_cost < reached[s].path_cost:
                reached[s] = child
                frontier.add(child)
    return failure, nos_explorados

def g(n): return n.path_cost

def astar_search(problem, h=None):
    """Search nodes with minimum f(n) = g(n) + h(n)."""
    h = h or problem.h
    return best_first_search(problem, f=lambda n: g(n) + h(n))


# |=====================|
# | Modelo da Cafeteria |
# |=====================|

class Estado:
    def __init__(self, garcom_pos, garcom_carga, bebidas, limpeza, bebidas_prontas):
        self.garcom_pos = garcom_pos
        self.garcom_carga = garcom_carga
        self.bebidas = bebidas
        self.limpeza = limpeza
        self.bebidas_prontas = bebidas_prontas

    def __str__(self):
        return f"pos={self.garcom_pos} | carga={self.garcom_carga} | bebidas={self.bebidas} | limpeza={self.limpeza} | prontas ={self.bebidas_prontas}"

    def __eq__(self, other):
        return (self.garcom_pos == other.garcom_pos and
                self.garcom_carga == other.garcom_carga and
                self.bebidas == other.bebidas and
                self.limpeza == other.limpeza and
                self.bebidas_prontas == other.bebidas_prontas)

    def __hash__(self):
        return hash((self.garcom_pos, tuple(self.garcom_carga), tuple(self.bebidas), tuple(self.limpeza), tuple(self.bebidas_prontas)))

    def mover(self, destino):
        if self.garcom_pos != destino:
            return Estado(destino, self.garcom_carga, self.bebidas, self.limpeza, self.bebidas_prontas)
        
    def preparar_bebida(self):
        pedido_frias = 0
        pedido_quentes = 0
        # Contabilizar pedidos de bebidas
        for pedido_drink in self.bebidas:
            pedido_frias += pedido_drink[0]
            pedido_quentes += pedido_drink[1]
        # Subtraindo bebidas prontas
        pedido_frias -= self.bebidas_prontas[0]
        pedido_quentes -= self.bebidas_prontas[1]
        # Fazer as bebidas
        if pedido_frias > 0:
            total_bebidas_prontas = copy.copy(self.bebidas_prontas)
            total_bebidas_prontas[0] += 1
            pedido_frias -= 1
            return Estado(self.garcom_pos, self.garcom_carga, self.bebidas, self.limpeza, total_bebidas_prontas)
        elif pedido_quentes > 0:
            total_bebidas_prontas = copy.copy(self.bebidas_prontas)
            total_bebidas_prontas[1] += 1
            pedido_quentes -= 1
            return Estado(self.garcom_pos, self.garcom_carga, self.bebidas, self.limpeza, total_bebidas_prontas)

    def pegar_bebida(self):
        if self.garcom_pos == 'bar' and self.garcom_carga == [0,0]:
            if self.bebidas_prontas[0] > 0:
                nova_carga = copy.copy(self.garcom_carga)
                nova_carga[0] += 1
                total_bebidas_prontas = copy.copy(self.bebidas_prontas)
                total_bebidas_prontas[0] -= 1
                return Estado(self.garcom_pos, nova_carga, self.bebidas, self.limpeza, total_bebidas_prontas)
            elif self.bebidas_prontas[1] > 0:
                nova_carga = copy.copy(self.garcom_carga)
                nova_carga[1] += 1
                total_bebidas_prontas = copy.copy(self.bebidas_prontas)
                total_bebidas_prontas[1] -= 1
                return Estado(self.garcom_pos, nova_carga, self.bebidas, self.limpeza, total_bebidas_prontas)

    def servir_bebida(self):
        if self.garcom_carga != [0,0] and self.garcom_pos != 'bar':
            mesa = int(self.garcom_pos[4])
            frias, quentes = self.bebidas[mesa-1]
            if frias > 0 and self.garcom_carga[0] > 0:
                novas_bebidas = copy.copy(self.bebidas)
                novas_bebidas[mesa-1] = (frias - 1, quentes)
                nova_carga = copy.copy(self.garcom_carga)
                nova_carga[0] -= 1
                return Estado(self.garcom_pos, nova_carga, novas_bebidas, self.limpeza, self.bebidas_prontas)
            elif quentes > 0 and self.garcom_carga[1] > 0:
                novas_bebidas = copy.copy(self.bebidas)
                novas_bebidas[mesa-1] = (frias, quentes - 1)
                nova_carga = copy.copy(self.garcom_carga)
                nova_carga[1] -= 1
                return Estado(self.garcom_pos, nova_carga, novas_bebidas, self.limpeza, self.bebidas_prontas)

    def limpar(self):
        if self.garcom_carga == [0,0] and self.garcom_pos != 'bar':
            mesa = int(self.garcom_pos[4])
            if self.limpeza[mesa - 1] != 0:
                nova_limpeza = copy.copy(self.limpeza)
                nova_limpeza[mesa - 1] = 0
                return Estado(self.garcom_pos, self.garcom_carga, self.bebidas, nova_limpeza, self.bebidas_prontas)


class CafeteriaProblem(Problem):
    def actions(self, state):
        return ['pegar_bebida', 'servir_bebida', 'limpar',
                'mover_bar', 'mover_mesa1', 'mover_mesa2',
                'mover_mesa3', 'mover_mesa4', 'preparar_bebida']

    def result(self, state, action):
        if action == 'pegar_bebida':    return state.pegar_bebida()
        if action == 'servir_bebida':   return state.servir_bebida()
        if action == 'limpar':          return state.limpar()
        if action == 'mover_bar':       return state.mover('bar')
        if action == 'mover_mesa1':     return state.mover('mesa1')
        if action == 'mover_mesa2':     return state.mover('mesa2')
        if action == 'mover_mesa3':     return state.mover('mesa3')
        if action == 'mover_mesa4':     return state.mover('mesa4')
        if action == 'preparar_bebida': return state.preparar_bebida()

    def is_goal(self, state):
        if state is None:
            return False
        return state.bebidas == [(0,0), (0,0), (0,0), (0,0)] and state.limpeza == [0, 0, 0, 0]


# |========================|
# | Definindo os Problemas |
# |========================|

#problema1 = Estado('bar', None, [0, 2, 0, 0], [0, 0, 1, 1])
problema1 = Estado('bar', [0,0], [(0,0), (2,0), (0,0), (0,0)], [0, 0, 1, 1], [0,0])
problema2 = Estado('bar', [0,0], [(0,0), (0,0), (2,2), (0,0)], [1, 0, 0, 0], [0,0])
problema3 = Estado('bar', [0,0], [(0,2), (0,0), (0,0), (0,2)], [0, 0, 1, 0], [0,0])
problema4 = Estado('bar', [0,0], [(2,0), (0,0), (0,4), (2,0)], [0, 1, 0, 0], [0,0])
p1 = CafeteriaProblem(initial=problema1)
p2 = CafeteriaProblem(initial=problema2)
p3 = CafeteriaProblem(initial=problema3)
p4 = CafeteriaProblem(initial=problema4)

selecionado = p1

print("=== BFS ===")
inicio = time.time()
resultado, nos = breadth_first_search(selecionado)
fim = time.time()
print(f"Estado final: {resultado.state}")
print(f"Nós explorados: {nos}")
print("Ações:", path_actions(resultado))
print(f"Tempo: {fim - inicio:.4f}s")

# print("\n=== Depth First ===")
# inicio = time.time()
# resultado, nos = depth_first_recursive_search(selecionado)
# fim = time.time()
# print(f"Estado final: {resultado.state}")
# print(f"Nós explorados: {nos}")
# print("Ações:", path_actions(resultado))
# print(f"Tempo: {fim - inicio:.4f}s")

# print("\n=== Iterative Deepening ===")
# inicio = time.time()
# resultado, nos = iterative_deepening_search(selecionado)
# fim = time.time()
# print(f"Estado final: {resultado.state}")
# print(f"Nós explorados: {nos}")
# print("Ações:", path_actions(resultado))
# print(f"Tempo: {fim - inicio:.4f}s")

print("=== A* ===")
inicio = time.time()
resultado, nos = astar_search(selecionado)
fim = time.time()
print(f"Estado final: {resultado.state}")
print(f"Nós explorados: {nos}")
print("Ações:", path_actions(resultado))
print(f"Tempo: {fim - inicio:.4f}s")