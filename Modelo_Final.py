import math
import sys
import copy
import heapq
import time
import threading
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
# | (com paralelismo)   |
# |=====================|

class Estado:
    def __init__(self, garcom_pos, garcom_carga, bebidas, limpeza, bebidas_prontas,
                 usa_bandeja=False, t_barista=0.0, t_garcom=0.0):
        self.garcom_pos      = garcom_pos
        self.garcom_carga    = garcom_carga       # [frias, quentes] carregadas
        self.bebidas         = bebidas            # [(frias, quentes), ...] por mesa
        self.limpeza         = limpeza            # [0/1, ...] por mesa
        self.bebidas_prontas = bebidas_prontas    # [frias, quentes] prontas no balcão
        self.usa_bandeja     = usa_bandeja
        self.t_barista       = t_barista          # tempo em que o barista fica livre
        self.t_garcom        = t_garcom           # tempo em que o garçom fica livre

    def __str__(self):
        bandeja = " [BANDEJA]" if self.usa_bandeja else ""
        return (f"pos={self.garcom_pos}{bandeja} | carga={self.garcom_carga} | "
                f"bebidas={self.bebidas} | limpeza={self.limpeza} | "
                f"prontas={self.bebidas_prontas} | "
                f"t_barista={self.t_barista:.1f} t_garcom={self.t_garcom:.1f}")

    def __eq__(self, other):
        return (self.garcom_pos      == other.garcom_pos      and
                self.garcom_carga    == other.garcom_carga    and
                self.bebidas         == other.bebidas         and
                self.limpeza         == other.limpeza         and
                self.bebidas_prontas == other.bebidas_prontas and
                self.usa_bandeja     == other.usa_bandeja     and
                round(self.t_barista, 4) == round(other.t_barista, 4) and
                round(self.t_garcom,  4) == round(other.t_garcom,  4))

    def __hash__(self):
        # Multiplicamos por 2 e arredondamos para int para evitar problemas
        # com ponto flutuante (todos os tempos são múltiplos de 0.5)
        return hash((self.garcom_pos,
                     tuple(self.garcom_carga),
                     tuple(self.bebidas),
                     tuple(self.limpeza),
                     tuple(self.bebidas_prontas),
                     self.usa_bandeja,
                     round(self.t_barista * 2),
                     round(self.t_garcom  * 2)))

    # --- Ações puras do estado (sem atualizar tempos) ---
    # Os tempos são atualizados pelo CafeteriaProblem.result()

    def mover(self, destino):
        if self.garcom_pos != destino:
            return Estado(destino, self.garcom_carga, self.bebidas,
                          self.limpeza, self.bebidas_prontas, self.usa_bandeja,
                          self.t_barista, self.t_garcom)

    def pegar_bandeja(self):
        if (self.garcom_pos == 'bar'
                and not self.usa_bandeja
                and self.garcom_carga == [0, 0]):
            return Estado(self.garcom_pos, self.garcom_carga, self.bebidas,
                          self.limpeza, self.bebidas_prontas, True,
                          self.t_barista, self.t_garcom)

    def devolver_bandeja(self):
        if (self.garcom_pos == 'bar'
                and self.usa_bandeja
                and self.garcom_carga == [0, 0]):
            return Estado(self.garcom_pos, self.garcom_carga, self.bebidas,
                          self.limpeza, self.bebidas_prontas, False,
                          self.t_barista, self.t_garcom)

    def pegar_bebida(self):
        if self.garcom_pos != 'bar':
            return None
        capacidade = 3 if self.usa_bandeja else 1
        if sum(self.garcom_carga) >= capacidade:
            return None
        if self.bebidas_prontas[0] > 0:
            nova_carga = copy.copy(self.garcom_carga)
            nova_carga[0] += 1
            prontas = copy.copy(self.bebidas_prontas)
            prontas[0] -= 1
            return Estado(self.garcom_pos, nova_carga, self.bebidas,
                          self.limpeza, prontas, self.usa_bandeja,
                          self.t_barista, self.t_garcom)
        elif self.bebidas_prontas[1] > 0:
            nova_carga = copy.copy(self.garcom_carga)
            nova_carga[1] += 1
            prontas = copy.copy(self.bebidas_prontas)
            prontas[1] -= 1
            return Estado(self.garcom_pos, nova_carga, self.bebidas,
                          self.limpeza, prontas, self.usa_bandeja,
                          self.t_barista, self.t_garcom)

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
                          self.limpeza, self.bebidas_prontas, self.usa_bandeja,
                          self.t_barista, self.t_garcom)
        elif quentes > 0 and self.garcom_carga[1] > 0:
            novas_bebidas = list(self.bebidas)
            novas_bebidas[mesa - 1] = (frias, quentes - 1)
            nova_carga = copy.copy(self.garcom_carga)
            nova_carga[1] -= 1
            return Estado(self.garcom_pos, nova_carga, novas_bebidas,
                          self.limpeza, self.bebidas_prontas, self.usa_bandeja,
                          self.t_barista, self.t_garcom)

    def limpar(self):
        if (self.garcom_carga == [0, 0]
                and self.garcom_pos != 'bar'
                and not self.usa_bandeja):
            mesa = int(self.garcom_pos[4])
            if self.limpeza[mesa - 1] != 0:
                nova_limpeza = copy.copy(self.limpeza)
                nova_limpeza[mesa - 1] = 0
                return Estado(self.garcom_pos, self.garcom_carga, self.bebidas,
                              nova_limpeza, self.bebidas_prontas, self.usa_bandeja,
                              self.t_barista, self.t_garcom)

    def preparar_bebida(self):
        pedido_frias   = sum(f for f, q in self.bebidas) - self.bebidas_prontas[0] - self.garcom_carga[0]
        pedido_quentes = sum(q for f, q in self.bebidas) - self.bebidas_prontas[1] - self.garcom_carga[1]
        if pedido_frias > 0:
            prontas = copy.copy(self.bebidas_prontas)
            prontas[0] += 1
            return Estado(self.garcom_pos, self.garcom_carga, self.bebidas,
                          self.limpeza, prontas, self.usa_bandeja,
                          self.t_barista, self.t_garcom)
        elif pedido_quentes > 0:
            prontas = copy.copy(self.bebidas_prontas)
            prontas[1] += 1
            return Estado(self.garcom_pos, self.garcom_carga, self.bebidas,
                          self.limpeza, prontas, self.usa_bandeja,
                          self.t_barista, self.t_garcom)


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

    ACOES_BARISTA = {'preparar_bebida'}
    ACOES_GARCOM  = {'pegar_bandeja', 'devolver_bandeja', 'pegar_bebida',
                     'servir_bebida', 'limpar',
                     'mover_bar', 'mover_mesa1', 'mover_mesa2',
                     'mover_mesa3', 'mover_mesa4'}

    def actions(self, state):
        return list(self.ACOES_BARISTA) + list(self.ACOES_GARCOM)

    def _distancia(self, origem, destino):
        return (self.distancias.get((origem, destino))
                or self.distancias.get((destino, origem)))

    def _duracao_garcom(self, s, a, s1):
        """Duração real de uma ação do garçom."""
        if a.startswith('mover'):
            dist = self._distancia(s.garcom_pos, s1.garcom_pos)
            return dist if s.usa_bandeja else dist / 2
        elif a == 'limpar':
            return self.tamanho_mesas[s.garcom_pos] * 2
        else:
            return 1  # pegar_bebida, servir_bebida, pegar_bandeja, devolver_bandeja

    def result(self, state, action):
        """
        Aplica a ação e atualiza o relógio do robô correspondente.

        Modelo de paralelismo:
          - Cada robô tem seu próprio relógio (t_barista, t_garcom).
          - Uma ação começa quando o robô fica livre (seu t_robot).
          - O relógio daquele robô avança pela duração da ação.
          - O outro robô continua no seu próprio tempo.
        """
        if action == 'preparar_bebida':
            s1 = state.preparar_bebida()
            if s1 is None:
                return None
            dur = 3 if s1.bebidas_prontas[0] > state.bebidas_prontas[0] else 5
            s1.t_barista = state.t_barista + dur
            # t_garcom não muda

        elif action.startswith('mover'):
            destino = action[6:]   # 'mover_bar' -> 'bar', 'mover_mesa1' -> 'mesa1'
            s1 = state.mover(destino)
            if s1 is None:
                return None
            dur = self._duracao_garcom(state, action, s1)
            s1.t_garcom = state.t_garcom + dur
            # t_barista não muda

        elif action == 'limpar':
            s1 = state.limpar()
            if s1 is None:
                return None
            dur = self._duracao_garcom(state, action, s1)
            s1.t_garcom = state.t_garcom + dur

        elif action == 'pegar_bandeja':
            s1 = state.pegar_bandeja()
            if s1 is None:
                return None
            s1.t_garcom = state.t_garcom + 1

        elif action == 'devolver_bandeja':
            s1 = state.devolver_bandeja()
            if s1 is None:
                return None
            s1.t_garcom = state.t_garcom + 1

        elif action == 'pegar_bebida':
            s1 = state.pegar_bebida()
            if s1 is None:
                return None
            s1.t_garcom = state.t_garcom + 1

        elif action == 'servir_bebida':
            s1 = state.servir_bebida()
            if s1 is None:
                return None
            s1.t_garcom = state.t_garcom + 1

        else:
            return None

        return s1

    def action_cost(self, s, a, s1):
        """
        Custo = variação no tempo total decorrido (max dos dois relógios).
        Se o robô que agiu era o mais lento, o custo pode ser 0
        (o outro robô ainda estava mais atrasado).
        """
        return max(s1.t_barista, s1.t_garcom) - max(s.t_barista, s.t_garcom)

    def is_goal(self, state):
        if state is None:
            return False
        return (state.bebidas      == [(0,0),(0,0),(0,0),(0,0)] and
                state.limpeza      == [0, 0, 0, 0]              and
                not state.usa_bandeja                           and
                state.garcom_carga == [0, 0])

    # --- Heurísticas ---

    def h(self, node):
        return 0

    def h1(self, node):
        """Contagem simples: bebidas pendentes + mesas sujas."""
        e = node.state
        return sum(f + q for f, q in e.bebidas) + sum(e.limpeza)

    def h2(self, node):
        """h1 + distância do garçom ao bar."""
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
        """Considera custos reais de preparo, limpeza e deslocamento."""
        e = node.state
        frias   = sum(f for f, q in e.bebidas)
        quentes = sum(q for f, q in e.bebidas)
        # O barista trabalha em paralelo, então o custo de preparo é
        # o máximo entre o tempo já gasto e o tempo necessário
        custo_preparo = max(frias * 3 + quentes * 5 - e.t_barista, 0)
        custo_limpeza = sum(
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
# | Visualização BFS        |
# |=========================|

ACOES_BARISTA = {'preparar_bebida'}

def exibir_plano(resultado, nome_problema):
    acoes   = path_actions(resultado)
    estados = path_states(resultado)

    tempo_total = max(estados[-1].t_barista, estados[-1].t_garcom)

    print(f"\n{'='*90}")
    print(f"  Problema: {nome_problema}  |  Tempo total: {tempo_total:.1f}  |  Ações: {len(acoes)}")
    print(f"{'='*90}")
    print(f"  Estado inicial: {estados[0]}")
    print(f"{'-'*90}")
    print(f"  {'Passo':<6} {'Robô':<10} {'Início':>8} {'Fim':>8}  {'Ação':<22} Estado resultante")
    print(f"{'-'*90}")

    for i, (acao, s_antes, s_depois) in enumerate(zip(acoes, estados[:-1], estados[1:]), start=1):
        if acao in ACOES_BARISTA:
            robo   = "Barista"
            t_ini  = s_antes.t_barista
            t_fim  = s_depois.t_barista
        else:
            robo   = "Garçom"
            t_ini  = s_antes.t_garcom
            t_fim  = s_depois.t_garcom
        print(f"  {i:<6} {robo:<10} {t_ini:>7.1f} {t_fim:>7.1f}  {acao:<22} {s_depois}")

    print(f"{'-'*90}")
    print(f"  t_barista final: {estados[-1].t_barista:.1f} | t_garcom final: {estados[-1].t_garcom:.1f} | "
          f"Tempo total (paralelo): {tempo_total:.1f}")
    print(f"{'='*90}\n")


# |=========================|
# | Funções auxiliares      |
# |=========================|

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


def depth_first_search(problem, timeout=30):
    """DFS iterativo com timeout para evitar estouro de recursão."""
    resultado_container = [failure, 0]
    stop_flag = threading.Event()

    def _rodar():
        nos_explorados = 0
        frontier = LIFOQueue([Node(problem.initial)])
        visited = set()
        while frontier:
            if stop_flag.is_set():
                return
            node = frontier.pop()
            nos_explorados += 1
            resultado_container[1] = nos_explorados
            if problem.is_goal(node.state):
                resultado_container[0] = node
                return
            if not is_cycle(node):
                for child in expand(problem, node):
                    if child.state not in visited:
                        visited.add(child.state)
                        frontier.append(child)
        resultado_container[0] = failure

    thread = threading.Thread(target=_rodar)
    thread.start()
    thread.join(timeout=timeout)

    if thread.is_alive():
        stop_flag.set()
        thread.join()
        return None, -1
    return resultado_container[0], resultado_container[1]


def depth_limited_search_stoppable(problem, limit, stop_flag):
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
        stop_flag.set()
        thread.join()
        return None, -1
    return resultado_container[0], resultado_container[1]


def tempo_total_resultado(resultado):
    """Extrai o tempo paralelo total do estado final."""
    if resultado in (failure, cutoff) or resultado is None:
        return float('inf')
    return max(resultado.state.t_barista, resultado.state.t_garcom)


# |=========================|
# | Testes                  |
# |=========================|

heuristicas = {
    'h0 (sem)': lambda p: p.h,
    'h1':       lambda p: p.h1,
    'h2':       lambda p: p.h2,
    'h3':       lambda p: p.h3,
    'h4':       lambda p: p.h4,
}

problemas_vis = {
    'Problema 1': p1,
    'Problema 2': p2,
    'Problema 3': p3,
    'Problema 4': p4,
}

problemas = {'p1': p1, 'p2': p2, 'p3': p3, 'p4': p4}

# --- Tabela comparativa: todos os algoritmos ---
W = 65

print("\n" + "=" * W)
print(f"{'BFS':<12} {'':12} {'Nós':>8} {'T. paralelo':>12} {'CPU':>10}")
print("-" * W)
for nome_p, problema in problemas.items():
    inicio = time.time()
    resultado, nos = breadth_first_search(problema)
    fim = time.time()
    print(f"{nome_p:<12} {'':12} {nos:>8} {tempo_total_resultado(resultado):>12.1f} {fim-inicio:>10.4f}s")
print()

print("=" * W)
print(f"{'DFS':<12} {'':12} {'Nós':>8} {'T. paralelo':>12} {'CPU':>10}")
print("-" * W)
for nome_p, problema in problemas.items():
    inicio = time.time()
    resultado, nos = depth_first_search(problema, timeout=180)
    fim = time.time()
    if nos == -1:
        print(f"{nome_p:<12} {'':12} {'N/A':>8} {'TIMEOUT':>12} {fim-inicio:>10.4f}s")
    else:
        print(f"{nome_p:<12} {'':12} {nos:>8} {tempo_total_resultado(resultado):>12.1f} {fim-inicio:>10.4f}s")
print()

print("=" * W)
print(f"{'IDS':<12} {'':12} {'Nós':>8} {'T. paralelo':>12} {'CPU':>10}")
print("-" * W)
for nome_p, problema in problemas.items():
    inicio = time.time()
    resultado, nos = iterative_deepening_search(problema, timeout=180)
    fim = time.time()
    if nos == -1:
        print(f"{nome_p:<12} {'':12} {'N/A':>8} {'TIMEOUT':>12} {fim-inicio:>10.4f}s")
    else:
        print(f"{nome_p:<12} {'':12} {nos:>8} {tempo_total_resultado(resultado):>12.1f} {fim-inicio:>10.4f}s")
print()

print("=" * W)
print(f"{'A*':<12} {'Heurística':<12} {'Nós':>8} {'T. paralelo':>12} {'CPU':>10}")
print("-" * W)
for nome_p, problema in problemas.items():
    for nome_h, h_func in heuristicas.items():
        inicio = time.time()
        resultado, nos = astar_search(problema, h=h_func(problema))
        fim = time.time()
        print(f"{nome_p:<12} {nome_h:<12} {nos:>8} {tempo_total_resultado(resultado):>12.1f} {fim-inicio:>10.4f}s")
    print()
print("=" * W)
