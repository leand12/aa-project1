
import itertools
import random as rdm
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt
from time import perf_counter
from typing import Tuple

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

        self.vertices[v] = w
        return self

    def add_edge(self, v1: Point, v2: Point):
        for v in (v1, v2):
            assert v in self.vertices, f"Vertice {v} does not exist"
        assert not v1 == v2, f"Vertices {v1} {v2} are equal"

        self.edges.setdefault(v1, set()).add(v2)
        self.edges.setdefault(v2, set()).add(v1)
        return self

    def read_graph(self, filename):
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

    def random_graph(self, size, seed=None):
        assert size <= 81, "Size must be lower than 81"

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

    def draw_graph(self):
        G = nx.Graph()

        for v, neighbours in self.edges.items():
            for n in neighbours:
                G.add_edge(v, n)

        low, *_, high = sorted(self.vertices.values())
        norm = mpl.colors.Normalize(
            vmin=low, vmax=high, clip=True)
        mapper = mpl.cm.ScalarMappable(
            norm=norm, cmap=mpl.cm.coolwarm)
        colors = [mapper.to_rgba(i)
                  for i in self.vertices.values()]
        positions = {x: x for x in G.nodes}

        sol_nodes = nx.draw_networkx_nodes(
            G,
            pos=positions,
            nodelist=self.solution[1][0],
            node_size=800,
            node_color="white"
        )
        sol_nodes.set_edgecolor('black')

        nx.draw(G,
                pos=positions,
                labels=self.vertices,
                nodelist=self.vertices,
                cmap=mpl.cm.summer,
                node_color=colors,
                font_weight='bold',
                font_color='whitesmoke',
                with_labels=True)

        plt.show()

    def subset_cost(self, subset):
        return sum(self.vertices[v] for v in subset)

    def is_dominating_subset(self, subset, graph):
        return all(self.edges[v] & subset for v in graph - subset)

    def heuristic(self, v: Point, ):
        return

    def exhaustive_combinations(self):
        sorted_weights = sorted(self.vertices.values())
        cumulative_min_weight = [sum(sorted_weights[:s + 1])
                                 for s in range(self.size)]

        for sub_size in range(1, self.size + 1):
            if cumulative_min_weight[sub_size - 1] > self.min_cost:
                # prune
                return

            for subset in itertools.combinations(range(self.size), sub_size):
                # cost = self.subset_cost(subset)
                yield None, subset

    def greedy_combinations(self):
        sorted_vertices = [x for x, _ in sorted(
            self.vertices.items(), key=lambda x: x[1])]
        n = len(sorted_vertices)
        queue = [(self.vertices[sorted_vertices[i]], (i,))
                 for i in reversed(range(n))]

        while queue:
            cost, index = queue.pop()

            if len(index) == n:
                continue

            subset = tuple(sorted_vertices[i] for i in index)

            if cost > self.min_cost:
                # prune
                return

            yield cost, subset

            for i in reversed(range(index[-1] + 1, n)):
                queue.append(
                    (cost + self.vertices[sorted_vertices[i]], (*index, i)))

    def astar_combinations(self):
        sorted_vertices = [x for x, _ in sorted(
            self.vertices.items(), key=lambda x: x[1])]
        n = len(sorted_vertices)
        queue = [(self.vertices[sorted_vertices[i]], (i,)) for i in range(n)]

        while queue:
            queue.sort(key=lambda x: -x[0])
            cost, index = queue.pop()

            subset = tuple(sorted_vertices[i] for i in index)

            if cost > self.min_cost:
                # prune
                return

            yield cost, subset

            for i in range(index[-1] + 1, n):
                queue.append(
                    (cost + self.vertices[sorted_vertices[i]], (*index, i)))

    def combinations(self, strategy):
        if strategy == 'exhaustive':
            return self.exhaustive_combinations()
        elif strategy == 'greedy':
            return self.greedy_combinations()
        elif strategy == 'astar':
            return self.astar_combinations()
        else:
            assert False, f"Unrecognized strategy \"{strategy}\""

    def search(self, strategy):
        graph = set(self.vertices)

        self.min_cost = self.subset_cost(graph)
        self.min_subsets = [graph]
        self.iter = 0
        start = perf_counter()

        for cost, subset in self.combinations(strategy):
            self.iter += 1

            subset = set(subset)
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


# g = SearchGraph().read_graph("graph1.txt")
g = SearchGraph().random_graph(20, 93446)

g.search('greedy')
print(g.solution, g.iter)
print(f"{g.exec_time:.6f}")

g.draw_graph()

g.search('astar')
print(g.solution, g.iter)
print(f"{g.exec_time:.6f}")

g.draw_graph()

g.search('exhaustive')
print(g.solution, g.iter)
print(f"{g.exec_time:.6f}")

g.draw_graph()
