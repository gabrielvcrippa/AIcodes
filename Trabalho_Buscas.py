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
    if node.parent is None: return []
    return path_actions(node.parent) + [node.action]

def path_states(node):
    if node in (cutoff, failure, None): return []
    return path_states(node.parent) + [node.state]

def is_cycle(node, k=30):
    def find_cycle(ancestor, k):
        return (ancestor is not None and k > 0 and
                (ancestor.state == node.state or find_cycle(ancestor.parent, k - 1)))
    return find_cycle(node.parent, k)

FIFOQueue = deque
LIFOQueue = list

class PriorityQueue:
    def __init__(self, items=(), key=lambda x: x):
        self.key = key
        self.items = []
        for item in items:
            self.add(item)
    def add(self, item):
        heapq.heappush(self.items, (self.key(item), item))
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

import threading

def iterative_deepening_search(problem, timeout=30):
    resultado_container = [failure, 0]
    stop_flag = threading.Event()

    def _rodar():
        total_nos = 0
        for limit in range(1, sys.maxsize):
            if stop_flag.is_set():
                return
            result, nos = depth_limited_search_stoppable(problem, limit, stop_flag)
            total_nos += nos
            resultado_container[1] = total_nos
            if result != cutoff:
                resultado_container[0] = result
                return
        resultado_container[0] = failure

    thread = threading.Thread(target=_rodar)
    thread.start()
    thread.join(timeout=timeout)

    if thread.is_alive():
        stop_flag.set()   # sinaliza para a thread parar
        thread.join()     # espera ela realmente terminar
        return None, -1
    return resultado_container[0], resultado_container[1]


def depth_limited_search_stoppable(problem, limit, stop_flag):
    """Versão do DLS que respeita a flag de parada."""
    nos_explorados = 0
    frontier = LIFOQueue([Node(problem.initial)])
    result = failure
    while frontier:
        if stop_flag.is_set():
            return cutoff, nos_explorados
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
    h = h or problem.h
    return best_first_search(problem, f=lambda n: g(n) + h(n))


# |=====================|
# | Modelo da Cafeteria |
# |=====================|

class Estado:
    def __init__(self, garcom_pos, garcom_carga, bebidas, limpeza, bebidas_prontas, usa_bandeja=False):
        self.garcom_pos     = garcom_pos
        self.garcom_carga   = garcom_carga      # [frias, quentes] carregadas
        self.bebidas        = bebidas           # [(frias, quentes), ...] por mesa
        self.limpeza        = limpeza           # [0/1, ...] por mesa
        self.bebidas_prontas = bebidas_prontas  # [frias, quentes] prontas no balcão
        self.usa_bandeja    = usa_bandeja       # bool: garçom está com a bandeja?

    def __str__(self):
        bandeja = " [BANDEJA]" if self.usa_bandeja else ""
        return (f"pos={self.garcom_pos}{bandeja} | carga={self.garcom_carga} | "
                f"bebidas={self.bebidas} | limpeza={self.limpeza} | prontas={self.bebidas_prontas}")

    def __eq__(self, other):
        return (self.garcom_pos      == other.garcom_pos and
                self.garcom_carga    == other.garcom_carga and
                self.bebidas         == other.bebidas and
                self.limpeza         == other.limpeza and
                self.bebidas_prontas == other.bebidas_prontas and
                self.usa_bandeja     == other.usa_bandeja)

    def __hash__(self):
        return hash((self.garcom_pos,
                     tuple(self.garcom_carga),
                     tuple(self.bebidas),
                     tuple(self.limpeza),
                     tuple(self.bebidas_prontas),
                     self.usa_bandeja))

    # --- Ações do garçom ---

    def mover(self, destino):
        if self.garcom_pos != destino:
            return Estado(destino, self.garcom_carga, self.bebidas,
                          self.limpeza, self.bebidas_prontas, self.usa_bandeja)

    def pegar_bandeja(self):
        """Pega a bandeja do balcão: deve estar no bar, mãos vazias e sem bandeja."""
        if (self.garcom_pos == 'bar'
                and not self.usa_bandeja
                and self.garcom_carga == [0, 0]):
            return Estado(self.garcom_pos, self.garcom_carga, self.bebidas,
                          self.limpeza, self.bebidas_prontas, True)

    def devolver_bandeja(self):
        """Devolve a bandeja ao balcão: deve estar no bar, com bandeja e mãos vazias."""
        if (self.garcom_pos == 'bar'
                and self.usa_bandeja
                and self.garcom_carga == [0, 0]):
            return Estado(self.garcom_pos, self.garcom_carga, self.bebidas,
                          self.limpeza, self.bebidas_prontas, False)

    def pegar_bebida(self):
        """
        Sem bandeja: capacidade máxima 1 bebida.
        Com bandeja: capacidade máxima 3 bebidas.
        """
        if self.garcom_pos != 'bar':
            return None
        capacidade = 3 if self.usa_bandeja else 1
        if sum(self.garcom_carga) >= capacidade:
            return None
        # Pega fria primeiro, depois quente
        if self.bebidas_prontas[0] > 0:
            nova_carga = copy.copy(self.garcom_carga)
            nova_carga[0] += 1
            prontas = copy.copy(self.bebidas_prontas)
            prontas[0] -= 1
            return Estado(self.garcom_pos, nova_carga, self.bebidas,
                          self.limpeza, prontas, self.usa_bandeja)
        elif self.bebidas_prontas[1] > 0:
            nova_carga = copy.copy(self.garcom_carga)
            nova_carga[1] += 1
            prontas = copy.copy(self.bebidas_prontas)
            prontas[1] -= 1
            return Estado(self.garcom_pos, nova_carga, self.bebidas,
                          self.limpeza, prontas, self.usa_bandeja)

    def servir_bebida(self):
        if self.garcom_carga == [0, 0] or self.garcom_pos == 'bar':
            return None
        mesa = int(self.garcom_pos[4])
        frias, quentes = self.bebidas[mesa - 1]
        if frias > 0 and self.garcom_carga[0] > 0:
            novas_bebidas = list(self.bebidas)
            novas_bebidas[mesa - 1] = (frias - 1, quentes)
            nova_carga = copy.copy(self.garcom_carga)
            nova_carga[0] -= 1
            return Estado(self.garcom_pos, nova_carga, novas_bebidas,
                          self.limpeza, self.bebidas_prontas, self.usa_bandeja)
        elif quentes > 0 and self.garcom_carga[1] > 0:
            novas_bebidas = list(self.bebidas)
            novas_bebidas[mesa - 1] = (frias, quentes - 1)
            nova_carga = copy.copy(self.garcom_carga)
            nova_carga[1] -= 1
            return Estado(self.garcom_pos, nova_carga, novas_bebidas,
                          self.limpeza, self.bebidas_prontas, self.usa_bandeja)

    def limpar(self):
        """Não pode limpar com bandeja."""
        if (self.garcom_carga == [0, 0]
                and self.garcom_pos != 'bar'
                and not self.usa_bandeja):
            mesa = int(self.garcom_pos[4])
            if self.limpeza[mesa - 1] != 0:
                nova_limpeza = copy.copy(self.limpeza)
                nova_limpeza[mesa - 1] = 0
                return Estado(self.garcom_pos, self.garcom_carga, self.bebidas,
                              nova_limpeza, self.bebidas_prontas, self.usa_bandeja)

    # --- Ação do barista ---

    def preparar_bebida(self):
        pedido_frias  = sum(f for f, q in self.bebidas) - self.bebidas_prontas[0]
        pedido_quentes = sum(q for f, q in self.bebidas) - self.bebidas_prontas[1]
        # Subtrai também o que o garçom já carrega
        pedido_frias  -= self.garcom_carga[0]
        pedido_quentes -= self.garcom_carga[1]
        if pedido_frias > 0:
            prontas = copy.copy(self.bebidas_prontas)
            prontas[0] += 1
            return Estado(self.garcom_pos, self.garcom_carga, self.bebidas,
                          self.limpeza, prontas, self.usa_bandeja)
        elif pedido_quentes > 0:
            prontas = copy.copy(self.bebidas_prontas)
            prontas[1] += 1
            return Estado(self.garcom_pos, self.garcom_carga, self.bebidas,
                          self.limpeza, prontas, self.usa_bandeja)


class CafeteriaProblem(Problem):
    distancias = {
        ('bar',   'mesa1'): 2,
        ('bar',   'mesa2'): 2,
        ('bar',   'mesa3'): 3,
        ('bar',   'mesa4'): 3,
        ('mesa1', 'mesa2'): 1,
        ('mesa1', 'mesa3'): 1,
        ('mesa1', 'mesa4'): 1,
        ('mesa2', 'mesa3'): 1,
        ('mesa2', 'mesa4'): 1,
        ('mesa3', 'mesa4'): 1,
    }

    tamanho_mesas = {
        'mesa1': 1,
        'mesa2': 1,
        'mesa3': 2,
        'mesa4': 1,
    }

    def actions(self, state):
        return ['preparar_bebida',
                'pegar_bandeja', 'devolver_bandeja',
                'pegar_bebida', 'servir_bebida', 'limpar',
                'mover_bar', 'mover_mesa1', 'mover_mesa2',
                'mover_mesa3', 'mover_mesa4']

    def result(self, state, action):
        if action == 'preparar_bebida':  return state.preparar_bebida()
        if action == 'pegar_bandeja':    return state.pegar_bandeja()
        if action == 'devolver_bandeja': return state.devolver_bandeja()
        if action == 'pegar_bebida':     return state.pegar_bebida()
        if action == 'servir_bebida':    return state.servir_bebida()
        if action == 'limpar':           return state.limpar()
        if action == 'mover_bar':        return state.mover('bar')
        if action == 'mover_mesa1':      return state.mover('mesa1')
        if action == 'mover_mesa2':      return state.mover('mesa2')
        if action == 'mover_mesa3':      return state.mover('mesa3')
        if action == 'mover_mesa4':      return state.mover('mesa4')

    def is_goal(self, state):
        if state is None:
            return False
        # Meta: todas as bebidas entregues, mesas limpas e bandeja devolvida
        return (state.bebidas      == [(0,0),(0,0),(0,0),(0,0)] and
                state.limpeza      == [0, 0, 0, 0] and
                not state.usa_bandeja and
                state.garcom_carga == [0, 0])

    def _distancia(self, origem, destino):
        return (self.distancias.get((origem, destino))
                or self.distancias.get((destino, origem)))

    def action_cost(self, s, a, s1):
        if a.startswith('mover'):
            dist = self._distancia(s.garcom_pos, s1.garcom_pos)
            # Com bandeja: velocidade 1 m/unidade → custo = dist
            # Sem bandeja: velocidade 2 m/unidade → custo = dist / 2
            return dist if s.usa_bandeja else dist / 2
        elif a == 'limpar':
            return self.tamanho_mesas[s.garcom_pos] * 2
        elif a == 'preparar_bebida':
            return 3 if s1.bebidas_prontas[0] > s.bebidas_prontas[0] else 5
        else:
            return 1  # pegar_bebida, servir_bebida, pegar_bandeja, devolver_bandeja

    # --- Heurísticas ---

    def h1(self, node):
        """Contagem simples: bebidas pendentes + mesas sujas."""
        e = node.state
        return sum(f + q for f, q in e.bebidas) + sum(e.limpeza)

    def h2(self, node):
        """h1 + distância do garçom ao bar (se houver bebidas)."""
        e = node.state
        bebidas = sum(f + q for f, q in e.bebidas)
        limpeza = sum(e.limpeza)
        dist_bar = 0
        if bebidas > 0 and e.garcom_pos != 'bar':
            dist_bar = self._distancia('bar', e.garcom_pos)
        return bebidas + limpeza + dist_bar

    def h3(self, node):
        """h2 + soma das distâncias bar→mesa para cada mesa com pedido."""
        e = node.state
        bebidas = sum(f + q for f, q in e.bebidas)
        limpeza = sum(e.limpeza)
        dist_bar = 0
        if bebidas > 0 and e.garcom_pos != 'bar':
            dist_bar = self._distancia('bar', e.garcom_pos)
        custo_entrega = sum(
            self._distancia('bar', f'mesa{i+1}')
            for i, pedido in enumerate(e.bebidas) if pedido != (0, 0)
        )
        return bebidas + limpeza + dist_bar + custo_entrega

    def h4(self, node):
        """Considera custos reais: preparo (3/5), limpeza (2×m²), deslocamento."""
        e = node.state
        frias   = sum(f for f, q in e.bebidas)
        quentes = sum(q for f, q in e.bebidas)
        custo_preparo  = frias * 3 + quentes * 5
        custo_limpeza  = sum(
            self.tamanho_mesas[f'mesa{i+1}'] * 2
            for i, suja in enumerate(e.limpeza) if suja
        )
        custo_desloc = 0
        if (frias + quentes) > 0 and e.garcom_pos != 'bar':
            custo_desloc = self._distancia('bar', e.garcom_pos) / 2
        return custo_preparo + custo_limpeza + custo_desloc


# |========================|
# | Definindo os Problemas |
# |========================|

problema1 = Estado('bar', [0,0], [(0,0),(2,0),(0,0),(0,0)], [0,0,1,1], [0,0])
problema2 = Estado('bar', [0,0], [(0,0),(0,0),(2,2),(0,0)], [1,0,0,0], [0,0])
problema3 = Estado('bar', [0,0], [(0,2),(0,0),(0,0),(0,2)], [0,0,1,0], [0,0])
problema4 = Estado('bar', [0,0], [(2,0),(0,0),(0,4),(2,0)], [0,1,0,0], [0,0])

p1 = CafeteriaProblem(initial=problema1)
p2 = CafeteriaProblem(initial=problema2)
p3 = CafeteriaProblem(initial=problema3)
p4 = CafeteriaProblem(initial=problema4)

# |=========================|
# | Teste automatizado      |
# |=========================|

heuristicas = {
    'h0 (sem)': lambda p: p.h,
    'h1':       lambda p: p.h1,
    'h2':       lambda p: p.h2,
    'h3':       lambda p: p.h3,
    'h4':       lambda p: p.h4,
}

problemas = {'p1': p1, 'p2': p2, 'p3': p3, 'p4': p4}

# --- BFS ---
print("=" * 55)
print(f"{'BFS':<12} {'':12} {'Nós':>8} {'Custo':>8} {'Tempo':>10}")
print("-" * 55)
for nome_p, problema in problemas.items():
    inicio = time.time()
    resultado, nos = breadth_first_search(problema)
    fim = time.time()
    custo = resultado.path_cost if resultado != failure else float('inf')
    print(f"{nome_p:<12} {'':12} {nos:>8} {custo:>8.1f} {fim-inicio:>10.4f}s")
print()

# --- DFS recursivo ---
print("=" * 55)
print(f"{'DFS':<12} {'':12} {'Nós':>8} {'Custo':>8} {'Tempo':>10}")
print("-" * 55)
for nome_p, problema in problemas.items():
    inicio = time.time()
    resultado, nos = depth_first_recursive_search(problema)
    fim = time.time()
    custo = resultado.path_cost if resultado not in (failure, cutoff) else float('inf')
    print(f"{nome_p:<12} {'':12} {nos:>8} {custo:>8.1f} {fim-inicio:>10.4f}s")
print()

# --- IDS ---
print("=" * 55)
print(f"{'IDS':<12} {'':12} {'Nós':>8} {'Custo':>8} {'Tempo':>10}")
print("-" * 55)
for nome_p, problema in problemas.items():
    inicio = time.time()
    resultado, nos = iterative_deepening_search(problema, timeout=30)
    fim = time.time()
    if nos == -1:
        print(f"{nome_p:<12} {'':12} {'N/A':>8} {'TIMEOUT':>8} {fim-inicio:>10.4f}s")
    else:
        custo = resultado.path_cost if resultado not in (failure, cutoff) else float('inf')
        print(f"{nome_p:<12} {'':12} {nos:>8} {custo:>8.1f} {fim-inicio:>10.4f}s")
print()

# --- A* com todas as heurísticas ---
print("=" * 55)
print(f"{'A*':<12} {'Heurística':<12} {'Nós':>8} {'Custo':>8} {'Tempo':>10}")
print("-" * 55)
for nome_p, problema in problemas.items():
    for nome_h, h_func in heuristicas.items():
        inicio = time.time()
        resultado, nos = astar_search(problema, h=h_func(problema))
        fim = time.time()
        custo = resultado.path_cost if resultado != failure else float('inf')
        print(f"{nome_p:<12} {nome_h:<12} {nos:>8} {custo:>8.1f} {fim-inicio:>10.4f}s")
    print()
print("=" * 55)
