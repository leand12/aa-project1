import argparse
import matplotlib.pyplot as plt
from typing import Tuple
from graph import SearchGraph

parser = argparse.ArgumentParser(
    usage="main.py [-h] (-f FILE | -r SEED) [-s N] [-a NAME] [-hr N]",
    description='Program that solves the Minimum-Weight Dominating Set Problem')

group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('-f', '--read-file', metavar='FILE', type=argparse.FileType('r'),
                   help='create a graph based on a correct format inside a file')
group.add_argument('-r', '--create-random', metavar='SEED', type=int,
                   help='create a random graph based on a seed')
parser.add_argument('-s', '--size', metavar='N', default=25, type=int, required=False,
                    help='the number of vertices of the graph (default: %(default)s)')
parser.add_argument('-a', '--algorithm', metavar='NAME', default='exhaustive', type=str, required=False,
                    choices=['exhaustive', 'branch_and_bound', 'greedy', 'astar', 'astar_heap'],
                    help='the number of vertices of the graph (default: %(default)s)')
parser.add_argument('-hr', '--heuristic', metavar='N', default=1, type=int, required=False,
                    help='the heuristic used by the Greedy and A-star approach: '
                    '(1) based on weights, (2) based on weights-degree, (default: %(default)s)')

args = vars(parser.parse_args())


def plot_basic_ops(sizes: Tuple[int, int], algos: Tuple[str, ...]):
    data = []
    print(f" Nodes & {' & '.join(a.title() for a in algos)}")

    for size in range(*sizes):
        g = SearchGraph().random_graph(size, 93446)
        data.append([])

        for algo in algos:
            g.search(algo)
            data[-1].append(g.basic_ops)

        print(f" {size}", end='')
        for d in data[-1]:
            print(f" & {d:d}", end='')
        print()

    # data.extend(
    #     [[15, 25387, 6474540],
    #     [13, 1779, 174436],
    #     [8, 557, 36456],
    #     [17, 3033, 1149016],
    #     [15, 5680, 1391841],
    #     [26, 28766, 25211935],
    #     [8, 2600, 2007327],
    #     [7, 520, 66711],
    #     [8, 236, 74518],
    #     [14, 11852, 15965871],
    #     [10, 5839, 3930550],
    #     [6, 627, 760098],
    #     [18, 1063, 5358577],
    #     [14, 9647, 33199095],
    #     [18, 5432, 39419863],
    #     [9, 7332, 46615613],
    #     [19, 15109, 54910659]]
    # )

    plt.yscale('log')
    plt.plot(range(*sizes), data)
    plt.title("Variation of the number of Operations with the Graph Size")
    plt.legend([a.title() for a in algos])
    plt.xlabel("Graph # of nodes")
    plt.ylabel("Basic operations log(ops)")
    plt.savefig("plots/basic_ops.png")
    plt.cla()
    plt.clf()


def plot_exec_s(sizes: Tuple[int, int], algos: Tuple[str, ...]):
    data = []
    print(f" nodes & {' & '.join(a.title() for a in algos)}")

    for size in range(*sizes):
        g = SearchGraph().random_graph(size, 93446)
        data.append([])

        for algo in algos:
            g.search(algo)
            data[-1].append(g.exec_)

        print(f" {size}", end='')
        for d in data[-1]:
            print(f" & {d*100:.3f}", end='')
        print()

    # data.extend(
    #     [[0.000379, 550.962854, 23.488071],
    #      [0.001075, 2.144161, 0.492601],
    #         [0.000152, 0.240921, 0.082189],
    #         [0.000272, 6.184916, 4.235634],
    #         [0.000279, 25.116781, 4.580443],
    #         [0.000524, 761.308984, 131.663930],
    #         [0.000223, 5.477160, 8.216070],
    #         [0.000191, 0.281841, 0.250342],
    #         [0.000217, 0.070878, 0.285507],
    #         [0.000314, 188.231096, 70.799879],
    #         [0.000269, 45.391021, 17.147758],
    #         [0.000152, 0.374619, 2.844985],
    #         [0.000420, 1.078891, 27.393365],
    #         [0.000382, 152.553328, 221.819140],
    #         [0.000454, 47.667561, 276.248185],
    #         [0.000244, 91.863296, 325.867944],
    #         [0.000511, 427.059270, 391.750860]]
    # )
    plt.yscale('log')
    plt.plot(range(*sizes), data)
    plt.title("Variation of the Execution  with the Graph Size")
    plt.legend([a.title() for a in algos])
    plt.xlabel("Graph # of nodes")
    plt.ylabel("Execution  log(s)")
    plt.savefig("plots/exec_s.png")
    plt.cla()
    plt.clf()


def plot_network_solutions(sizes: Tuple[int, int], algos: Tuple[str, ...]):

    for size in range(*sizes):
        g = SearchGraph().random_graph(size, 93446)

        fig, ax = plt.subplots(1, 3, num=1)
        fig.set_size_inches(21, 7)

        for i, algo in enumerate(algos):

            g.search(algo)
            print(g.solution, g.basic_ops, f"{g.exec_:.6f}")

            plt.sca(ax[i])
            g.draw_graph(ax[i])

            ax[i].set_title(
                f"Solution with {algo.title()} Search and size = {size}", fontsize=14)
            ax[i].set_xlabel(f"total cost = {g.solution[0]}", fontsize=12)
            ax[i].set_axis_on()
            ax[i].set_xlim(0, 10)
            ax[i].set_ylim(0, 10)
            ax[i].tick_params(left=True, bottom=True,
                              labelleft=True, labelbottom=True)

        fig.savefig(f"plots/g{size}.png", dpi=100)
        fig.clear()


if __name__ == '__main__':
    # g = SearchGraph().read_graph("graph1.txt")

    # plot_basic_ops((2, 46), ('greedy', 'astar_heap', 'exhaustive'))
    # plot_exec_s((2, 46), ('greedy', 'astar_heap', 'exhaustive'))
    # plot_network_solutions((2, 45), ('greedy', 'astar', 'exhaustive'))

    seed = args["create_random"]
    size = args["size"]
    algo = args["algorithm"]
    heur = args["heuristic"]
    
    if seed:
        g = SearchGraph().random_graph(size, seed)
    else:
        g = SearchGraph().read_graph(args["read_file"].name)

    g.search(algo, heur)
    g.draw_graph()

    print('Solution:', g.solution)
    print(f"({g.exec_time:.6f} seconds, {g.basic_ops} basic ops)")

    plt.title(
        f"Solution with {algo.title()} Search and size = {size}", fontsize=14)
    plt.xlabel(f"total cost = {g.solution[0]}", fontsize=12)
    plt.axis('on')
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    plt.tick_params(left=True, bottom=True,
                    labelleft=True, labelbottom=True)
    plt.show()
