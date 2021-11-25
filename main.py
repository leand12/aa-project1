
import random
import itertools
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt
from timeit import default_timer as timer


class SearchGraph:

    def __init__(self):
        self.vertices = {}
        self.edges = {}
        self.min_cost = None
        self.min_subsets = None
        self.time = None

    @property
    def size(self):
        return len(self.vertices)

    @property
    def solution(self):
        return self.min_cost, self.min_subsets

    def add_vertice(self, v: int, w: int):
        assert w >= 0, "Weight must be positve"
        assert v not in self.vertices, f"Vertice {v} already added"

        self.vertices[v] = w
        return self

    def add_edge(self, v1: int, v2: int):
        for v in (v1, v2):
            assert v in self.vertices, f"Vertice {v} does not exist"
        assert not v1 == v2, f"Vertices {v1} {v2} are equal"

        self.edges.setdefault(v1, set()).add(v2)
        self.edges.setdefault(v2, set()).add(v1)
        return self

    def read_graph(self, filename):

        try:
            file_ = open(filename, "r")
        except FileNotFoundError:
            print(f"ERR: File \"{filename}\" not found")

        l = 0
        while file_:
            line = file_.readline().strip()
            l += 1
            if not line:
                continue
            try:
                num_vertices = int(line)
            except:
                print(f"ERR: Expected a number on line {l}")

            for v in range(num_vertices):
                weight, *neighbours = file_.readline().strip().split()
                l += 1

                try:
                    weight = int(weight)
                    neighbours = [int(n) for n in neighbours]
                except:
                    print(f"ERR: Expected a number on line {l}")

                self.add_vertice(v, weight)
                for n in neighbours:
                    self.add_edge(v, n)
            break  # TODO: remove this
        return self

    def random_graph(self, size=15, seed=2):
        random.seed(seed)

        for v in range(size):
            self.add_vertice(v, random.randint(1, size//2))

        for v1 in range(size - 1):
            for _ in range(random.randint(1, size // 2)):
                v2 = random.randint(v1 + 1, size - 1)
                self.add_edge(v1, v2)

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

        nx.draw(G,
                nodelist=self.vertices,
                cmap=mpl.cm.summer,
                node_color=colors,
                font_weight='bold',
                font_color='white',
                with_labels=True)

        plt.show()

    def set_cost(self, subset):
        return sum(self.vertices[v] for v in subset)

    def is_dominating_set(self, subset, graph):
        return all(self.edges[v] & subset for v in graph - subset)

    def exhaustive_combinations(self):
        sorted_weights = sorted(self.vertices.values())
        cumulative_min_weight = [sum(sorted_weights[:s + 1])
                                 for s in range(self.size)]

        for sub_size in range(1, self.size + 1):
            if cumulative_min_weight[sub_size - 1] > self.min_cost:
                # prune
                return

            for subset in itertools.combinations(range(self.size), sub_size):
                # cost = self.set_cost(subset)
                yield None, subset

    def greedy_combinations(self):
        sorted_vertices = [x for x, _ in sorted(
            self.vertices.items(), key=lambda x: x[1])]
        n = len(sorted_vertices)
        queue = [(self.vertices[sorted_vertices[i]], (i,)) for i in reversed(range(n))]

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
                queue.append((cost + self.vertices[sorted_vertices[i]], (*index, i)))

    def greedy_combinations2(self):
        sorted_vertices = [x for x, _ in sorted(
            self.vertices.items(), key=lambda x: (x[1], len(self.edges[x[0]])))]
        n = len(sorted_vertices)
        queue = [(self.vertices[sorted_vertices[i]], (i,)) for i in reversed(range(n))]

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
                queue.append((cost + self.vertices[sorted_vertices[i]], (*index, i)))

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
        elif strategy == 'greedy2':
            return self.greedy_combinations2()
        elif strategy == 'astar':
            return self.astar_combinations()
        else:
            assert False, f"Unrecognized strategy \"{strategy}\""

    def search(self, strategy):
        graph = set(range(self.size))

        self.min_cost = self.set_cost(graph)
        self.min_subsets = [graph]
        self.iter = 0
        start = timer()

        for cost, subset in self.combinations(strategy):
            self.iter += 1

            subset = set(subset)
            if not self.is_dominating_set(subset, graph):
                continue

            if cost is None:
                cost = self.set_cost(subset)

            if cost < self.min_cost:
                self.min_cost = cost
                self.min_subsets = [subset]
            elif cost == self.min_cost:
                self.min_subsets.append(subset)

        self.time = timer() - start


# g = SearchGraph().read_graph("graph2.txt")

g = SearchGraph().random_graph(size=40, seed=93446)     # (280) 0.0002 0.0003 (65) 7.799954  14.140436
# g = SearchGraph().random_graph(size=40, seed=234)       # (280) 0.0002 0.0003 (65) 2.540548  80.498313
# g = SearchGraph().random_graph(size=25, seed=93302)     # (280) 0.0002 0.0003 (65) 12.649675 4.203344
# g = SearchGraph().random_graph(size=30, seed=93015)     # (280) 0.0002 0.0003 (65) 5.463316  20.374070
g = SearchGraph().random_graph(size=35, seed=3232)     # (280) 0.0002 0.0003 (65) 7.799954  14.140436
g = SearchGraph().random_graph(size=70, seed=32424)     # (280) 0.0002 0.0003 (65) 7.799954  14.140436
g = SearchGraph().random_graph(size=70, seed=13)     # (280) 0.0002 0.0003 (65) 7.799954  14.140436
g = SearchGraph().random_graph(size=100, seed=5)     # (280) 0.0002 0.0003 (65) 7.799954  14.140436

# g = SearchGraph().random_graph(size=16, seed=93446)

g.draw_graph()

g.search('greedy')
print(g.solution, g.iter)
print(f"{g.time:.6f}")

g.search('greedy2')
print(g.solution, g.iter)
print(f"{g.time:.6f}")

g.search('astar')
print(g.solution, g.iter)
print(f"{g.time:.6f}")

g.search('exhaustive')
print(g.solution, g.iter)
print(f"{g.time:.6f}")



g.draw_graph()

# http://webhome.cs.uvic.ca/~wendym/courses/445/14/445_heur.html
