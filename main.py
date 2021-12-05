import argparse
import matplotlib.pyplot as plt
from typing import Tuple
from graph import SearchGraph


def plot_basic_ops(sizes: Tuple[int, int], algos: Tuple[str, ...]):
    data = []

    for size in range(*sizes):
        g = SearchGraph().random_graph(size, 93446)
        data.append([])

        for algo in algos:
            g.search(algo)
            data[-1].append(g.basic_ops)

    plt.yscale('log')
    plt.plot(range(*sizes), data)
    plt.title("Variation of the number of Operations with the Graph Size")
    plt.legend([a.title() for a in algos])
    plt.xlabel("Graph # of nodes")
    plt.ylabel("Basic operations log(ops)")
    plt.savefig("plots/basic_ops.png")
    plt.cla()
    plt.clf()


def plot_exec_times(sizes: Tuple[int, int], algos: Tuple[str, ...]):
    data = []

    for size in range(*sizes):
        g = SearchGraph().random_graph(size, 93446)
        data.append([])

        for algo in algos:
            g.search(algo)
            data[-1].append(g.exec_time)

    plt.yscale('log')
    plt.plot(range(*sizes), data)
    plt.title("Variation of the Execution  with the Graph Size")
    plt.legend([a.title() for a in algos])
    plt.xlabel("Graph # of nodes")
    plt.ylabel("Execution  log(s)")
    plt.savefig("plots/exec_times.png")
    plt.cla()
    plt.clf()


def plot_network_solutions(sizes: Tuple[int, int], algos: Tuple[str, ...]):

    for size in range(*sizes):
        g = SearchGraph().random_graph(size, 93446)

        fig, ax = plt.subplots(1, 3, num=1)
        fig.set_size_inches(21, 7)

        for i, algo in enumerate(algos):

            g.search(algo)
            print(g.solution, g.basic_ops, f"{g.exec_time:.6f}")

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
    parser = argparse.ArgumentParser(
        description='Program that solves the Minimum-Weight Dominating Set Problem')

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-f', '--read-file', metavar='FILE', type=argparse.FileType('r'),
                       help='create a graph based on a correct format inside a file')
    group.add_argument('-r', '--create-random', metavar='SEED', type=int,
                       help='create a random graph based on a seed')
    parser.add_argument('-s', '--size', metavar='N', default=25, type=int, required=False,
                        help='the number of vertices of the graph (default: %(default)s)')
    parser.add_argument('-a', '--algorithm', metavar='NAME', default='exhaustive', type=str, required=False,
                        choices=['exhaustive', 'branch-and-bound',
                                 'greedy', 'astar', 'astar-heap'],
                        help='the number of vertices of the graph (default: %(default)s)')
    parser.add_argument('-hr', '--heuristic', metavar='N', default=1, type=int, required=False,
                        help='the heuristic used by the Greedy and A-star approach: '
                        '(1) based on weights, (2) based on weights-degree, (default: %(default)s)')

    args = vars(parser.parse_args())

    # g = SearchGraph().read_graph("graph1.txt")

    # plot_basic_ops((2, 46), ('greedy', 'astar-heap', 'exhaustive'))
    # plot_exec_times((2, 46), ('greedy', 'astar-heap', 'exhaustive'))
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
