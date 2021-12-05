import heapq
import itertools
import random as rdm
import networkx as nx
import matplotlib as mpl
from time import perf_counter
from typing import Tuple, Set, Iterable, Callable

Point = Tuple[int, int]


class SearchGraph:

    def __init__(self):
        self.vertices = {}
        self.edges = {}
        self.min_cost = None
        self.min_subsets = None
        self.exec_time = None

    @property
    def size(self):
        return len(self.vertices)

    @property
    def solution(self):
        return self.min_cost, self.min_subsets

    def add_vertice(self, v: Point, w: int):
        assert w >= 0, "Weight must be positive"
        assert isinstance(v, tuple), f"Vertice must be a tuple"
        assert v not in self.vertices, f"Vertice {v} already added"
        assert len(self.vertices) <= 81, "Limit of vertices exceeded"

        self.vertices[v] = w
        return self

    def add_edge(self, v1: Point, v2: Point):
        for v in (v1, v2):
            assert v in self.vertices, f"Vertice {v} does not exist"
        assert not v1 == v2, f"Vertices {v1} {v2} are equal"

        self.edges.setdefault(v1, set()).add(v2)
        self.edges.setdefault(v2, set()).add(v1)
        return self

    def read_graph(self, filename: str):
        self.vertices.clear()
        self.edges.clear()

        try:
            file_ = open(filename, "r")
        except FileNotFoundError:
            print(f"ERR: File \"{filename}\" not found")
            exit(1)

        l = 0
        while file_:
            line = file_.readline()
            l += 1
            if not line:
                continue
            try:
                num_vertices = int(line)
            except:
                print(f"ERR: Expected a number on line {l}")
                exit(1)

            vertices = []
            for v in range(num_vertices):
                try:
                    x, y, weight, *neighbours = [int(w)
                                                 for w in file_.readline().split()]
                    l += 1
                except:
                    print(f"ERR: Expected a number on line {l}")
                    exit(1)

                vertices.append((x, y))
                self.add_vertice((x, y), weight)
                for n in neighbours:
                    if not n in range(0, len(vertices)):
                        print(
                            f"ERR: Expected a number between [{0},{len(vertices)}[ on line {l}")
                        exit(1)
                    self.add_edge((x, y), vertices[n])
            break
        return self

    def random_graph(self, size: int, seed: int = None):
        assert 2 <= size <= 81, "Size must be between [2, 81]"

        self.vertices.clear()
        self.edges.clear()
        rdm.seed(seed)

        vertices = []
        for v in range(size):
            while True:
                x, y = (rdm.randint(1, 9), rdm.randint(1, 9))
                if not (x, y) in self.vertices:
                    break
            vertices.append((x, y))
            self.add_vertice((x, y), rdm.randint(1, 99))

        for v1 in range(size - 1):
            for _ in range(rdm.randint(1, size // 2)):
                v2 = rdm.randint(v1 + 1, size - 1)
                self.add_edge(vertices[v1], vertices[v2])

        return self

    def draw_graph(self, ax=None):
        G = nx.Graph()

        for v, neighbours in self.edges.items():
            for n in neighbours:
                G.add_edge(v, n)

        positions = {x: x for x in G.nodes}
        low, *_, high = sorted(self.vertices.values())
        norm = mpl.colors.Normalize(
            vmin=low, vmax=high, clip=True)
        mapper = mpl.cm.ScalarMappable(
            norm=norm, cmap=mpl.cm.coolwarm)
        colors = [mapper.to_rgba(i)
                  for i in self.vertices.values()]

        sol_vertices = {}
        sol_colors = []
        rem_vertices = {}
        rem_colors = []
        for i, (v, w) in enumerate(self.vertices.items()):
            if v in self.solution[1][0]:
                sol_vertices[v] = w
                sol_colors.append(colors[i])
            else:
                rem_vertices[v] = w
                rem_colors.append(colors[i])

        nx.draw_networkx_edges(
            G,
            pos=positions,
            nodelist=self.vertices,
            edge_color='gray',
            node_size=500,
            width=2,
            ax=ax)

        nx.draw_networkx_nodes(
            G,
            pos=positions,
            labels=sol_vertices,
            nodelist=sol_vertices,
            node_size=800,
            node_color=sol_colors,
            cmap=mpl.cm.summer,
            ax=ax)

        rem_nodes = nx.draw_networkx_nodes(
            G,
            pos=positions,
            labels=rem_vertices,
            nodelist=rem_vertices,
            node_size=500,
            node_color='white',
            cmap=mpl.cm.summer,
            ax=ax)
        rem_nodes.set_edgecolors(rem_colors)
        rem_nodes.set_linewidth(2)

        nx.draw_networkx_labels(
            G,
            pos=positions,
            labels=self.vertices,
            font_weight='bold',
            font_color='black',
            with_labels=True)

    def subset_cost(self, subset: Iterable[Point]):
        return sum(self.vertices[v] for v in subset)

    def is_dominating_subset(self, subset: Set[Point], graph: Set[Point]):
        return all(self.edges[v] & subset for v in graph - subset)

    def heuristic(self):
        return [v for v, _ in sorted(
            self.vertices.items(), key=lambda x: x[1])]

    def heuristic2(self):
        return [v for v, _ in sorted(
            self.vertices.items(), key=lambda x: x[1] - len(self.edges(x[0])))]

    def exhaustive_combinations(self, with_bnb=True):
        if with_bnb:
            sorted_weights = sorted(self.vertices.values())
            cumulative_min_weight = [sum(sorted_weights[:s + 1])
                                     for s in range(self.size)]
        vertices = list(self.vertices.keys())

        for sub_size in range(1, self.size + 1):
            if with_bnb and cumulative_min_weight[sub_size - 1] > self.min_cost:
                # prune: global minimum already found
                return

            for index in itertools.combinations(range(self.size), sub_size):
                subset = set(vertices[i] for i in index)
                # it is inefficient to calculate cost here
                yield None, subset

    def greedy_combinations(self, heuristic: Callable):
        sorted_vertices = heuristic()
        n = len(sorted_vertices)
        queue = [(self.vertices[sorted_vertices[i]], (i,))
                 for i in reversed(range(n))]

        while queue:
            cost, index = queue.pop()

            subset = set(sorted_vertices[i] for i in index)

            if cost > self.min_cost:
                # prune: local minimum found
                return

            yield cost, subset

            for i in reversed(range(index[-1] + 1, n)):
                queue.append(
                    (cost + self.vertices[sorted_vertices[i]], (*index, i)))

    def astar_combinations(self, heuristic: Callable, with_heap: bool = True):
        sorted_vertices = heuristic()
        n = len(sorted_vertices)
        queue = [(self.vertices[sorted_vertices[i]], (i,)) for i in range(n)]

        while queue:
            if with_heap:
                cost, index = heapq.heappop(queue)
            else:
                # inversed order to pop the end of the list
                # and avoid more computational costs
                queue.sort(key=lambda x: -x[0])
                cost, index = queue.pop()

            subset = set(sorted_vertices[i] for i in index)

            if cost > self.min_cost:
                # prune: global minimum found
                return

            yield cost, subset

            for i in range(index[-1] + 1, n):
                node = (cost + self.vertices[sorted_vertices[i]], (*index, i))
                if with_heap:
                    heapq.heappush(queue, node)
                else:
                    queue.append(node)

    def combinations(self, strategy: str, heuristic: str):
        heuristic = self.heuristic2 if heuristic == 2 else self.heuristic

        if strategy == 'exhaustive':
            return self.exhaustive_combinations(False)
        elif strategy == 'branch-and-bound':
            return self.exhaustive_combinations()
        elif strategy == 'greedy':
            return self.greedy_combinations(heuristic)
        elif strategy == 'astar':
            return self.astar_combinations(heuristic, False)
        elif strategy == 'astar-heap':
            return self.astar_combinations(heuristic)
        else:
            assert False, f"Unrecognized strategy \"{strategy}\""

    def search(self, strategy: str, heuristic: int = 1):
        assert len(self.vertices) >= 2, "Not enough vertices to search"
        graph = set(self.vertices)

        self.min_cost = self.subset_cost(graph)
        self.min_subsets = [graph]
        self.basic_ops = 0
        start = perf_counter()

        for cost, subset in self.combinations(strategy, heuristic):
            self.basic_ops += 1

            if not self.is_dominating_subset(subset, graph):
                continue

            if cost is None:
                cost = self.subset_cost(subset)

            if cost < self.min_cost:
                self.min_cost = cost
                self.min_subsets = [subset]
            elif cost == self.min_cost:
                self.min_subsets.append(subset)

        self.exec_time = perf_counter() - start
