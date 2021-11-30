
import itertools
import random as rdm
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt
from time import perf_counter
import time
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
        
        for i, v in enumerate(self.vertices):
            r, g, b, a = colors[i]
            if v not in self.solution[1][0]:
                colors[i] = (r, g, b, .6)

        sol_vertices = []
        sol_colors = []
        for i, v in enumerate(self.vertices):
            if v in self.solution[1][0]:
                sol_vertices.append(v)
                sol_colors.append(colors[i])

        sol_nodes = nx.draw_networkx_nodes(
            G,
            pos=positions,
            nodelist=sol_vertices,
            node_size=600,
            node_color="white",
            ax=ax
        )
        # sol_nodes.set_edgecolors(sol_colors)
        sol_nodes.set_edgecolors('limegreen')
        # sol_nodes.set_linestyle('dashed')
        sol_nodes.set_linewidth(2)

        nx.draw(G,
                pos=positions,
                labels=self.vertices,
                nodelist=self.vertices,
                cmap=mpl.cm.summer,
                node_color=colors,
                edge_color='gray',
                font_weight='bold',
                font_color='white',
                with_labels=True,
                ax=ax)

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
        
        vertices = list(self.vertices.keys())

        for sub_size in range(1, self.size + 1):
            if cumulative_min_weight[sub_size - 1] > self.min_cost:
                # prune
                return

            for index in itertools.combinations(range(self.size), sub_size):
                # cost = self.subset_cost(subset)
                subset = set(vertices[i] for i in index)
                yield None, subset

    def greedy_combinations(self):
        sorted_vertices = [x for x, _ in sorted(
            self.vertices.items(), key=lambda x: x[1])]
        n = len(sorted_vertices)
        queue = [(self.vertices[sorted_vertices[i]], (i,))
                 for i in reversed(range(n))]

        while queue:
            cost, index = queue.pop()

            subset = set(sorted_vertices[i] for i in index)

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

            subset = set(sorted_vertices[i] for i in index)

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
        assert len(self.vertices) >= 2, "Not enough vertices to search"
        graph = set(self.vertices)

        self.min_cost = self.subset_cost(graph)
        self.min_subsets = [graph]
        self.iter = 0
        start = perf_counter()

        for cost, subset in self.combinations(strategy):
            self.iter += 1

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

for size in range(2, 40):
    g = SearchGraph().random_graph(size, 93446)

    fig, ax = plt.subplots(1, 3, num=1)
    fig.set_size_inches(21, 7)

    for i, algo in enumerate(('greedy', 'astar', 'exhaustive')):

        g.search(algo)
        print(g.solution, g.iter)
        print(f"{g.exec_time:.6f}")

        plt.sca(ax[i])
        g.draw_graph(ax[i])

        ax[i].set_title(f"Solution with {algo.capitalize()} Search and size = {size}", fontsize=14)
        ax[i].set_xlabel(f"total cost = {g.solution[0]}", fontsize=12)
        ax[i].set_axis_on()
        ax[i].set_xlim(0, 10)
        ax[i].set_ylim(0, 10)
        ax[i].tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)

    fig.savefig(f"plots/g{size}.png", dpi=100)
    fig.clear()
