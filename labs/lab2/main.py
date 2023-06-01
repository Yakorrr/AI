import random
from time import time_ns
from collections import namedtuple

import matplotlib.pyplot as plt
import psutil

from stack import Stack
from priority_queue import PriorityQueue
from queue import Queue

plt.rcParams['figure.figsize'] = (6, 6)


class NQueensState:
    def __init__(self, N, queens=None):
        self.N = N
        self.queens = queens.copy() if queens else self.random_state()
        self.num_conflicts = self.conflicts()

    def __lt__(self, other):
        if self is other: return True
        if other is None: return False
        if not isinstance(other, NQueensState): return False

        return self.conflicts() >= other.conflicts()

    def __hash__(self):
        return hash(tuple(self.queens))

    def __str__(self):
        return f'{self.queens} <{self.conflicts()}>'

    def __repr__(self):
        return f'NQueensState(queens={self.queens})'

    def random_state(self):
        queens = list(range(1, self.N + 1))
        random.shuffle(queens)

        return queens

    def conflicts(self, queens=None):
        if queens is None: queens = self.queens[:]

        return sum([abs(queens[j] - queens[i]) == j - i
                    for i in range(self.N - 1)
                    for j in range(i + 1, self.N)])

    def neighbors(self, queens=None):
        if queens is None: queens = self.queens[:]

        N = self.N

        for i in range(N - 1):
            for j in range(i + 1, N):
                neighbor = NQueensState(N, queens=queens)
                neighbor.queens[i], neighbor.queens[j] = neighbor.queens[j], neighbor.queens[i]
                yield neighbor

    def random_neighbor(self, queens=None):
        if queens is None: queens = self.queens[:]

        N = self.N

        i = random.randint(0, N - 2)
        j = random.randint(i + 1, N - 1)

        neighbor = NQueensState(N, queens=queens)
        neighbor.queens[i], neighbor.queens[j] = neighbor.queens[j], neighbor.queens[i]
        neighbors = [(neighbor, 1)]

        return neighbors

    def plot(self, fc='darkgoldenrod', line_color='seagreen', line_alpha=0.8):
        N = self.N
        figsize = plt.rcParams['figure.figsize']
        figure = plt.figure(figsize=(6, 6))
        ax = figure.add_subplot(1, 1, 1)

        border = plt.Rectangle((0, -N), N, N, ec=fc, fc='w', alpha=0.35)
        ax.add_patch(border)

        # draw chess board
        for i in range(N):
            for j in range(N):
                alpha = 0.35 if (i + j) % 2 == 0 else 0.1
                cell = plt.Rectangle((i, -j - 1), 1, 1, fc=fc, alpha=alpha)
                ax.add_patch(cell)

        # place queens on chess board
        y = 0.5

        for position in self.queens:
            x = position - 0.5
            fs = max(1, figsize[0] * 50 // N)
            y -= 1
            ax.text(x, y, '♛', color='k', fontsize=fs, ha='center', va='center')

        ax.axis('square')
        ax.axis('off')
        ax.set_title("Conflicts = {}".format(self.conflicts()), fontsize=18)

        # show conflicts
        for i in range(N - 1):
            row_i = self.queens[i]

            for j in range(i + 1, N):
                row_j = self.queens[j]

                if abs(row_i - row_j) == j - i:
                    x1, x2 = row_i - 0.5, row_j - 0.5
                    y1, y2 = -i - 0.5, -j - 0.5

                    line = plt.Line2D((x1, x2), (y1, y2), lw=3, ls='-', color=line_color, alpha=line_alpha)
                    plt.plot(x1, y1, lw=3, ls='', marker='o', color=line_color, alpha=line_alpha)
                    plt.plot(x2, y2, lw=3, ls='', marker='o', color=line_color, alpha=line_alpha)
                    ax.add_line(line)

        plt.show()

    def LDFS(self, initial_state, depth_limit, time_limit, memory_limit):
        start_time = time_ns()
        visited_states = set()
        stack = Stack()
        stack.push((initial_state, 0))
        process = psutil.Process()
        max_nodes, generated_nodes = 0, 0

        while not stack.is_empty():
            current_state, current_depth = stack.pop()
            current_time = (time_ns() - start_time) / 1e9
            max_nodes = max(max_nodes, stack.length())
            start_length = stack.length()

            current_memory = process.memory_info().rss / (1024 ** 2)

            if current_time >= time_limit or current_memory >= memory_limit:
                return None

            if current_depth == depth_limit:
                continue

            for neighbor in self.neighbors():
                if tuple(neighbor.queens) not in visited_states:
                    visited_states.add(tuple(neighbor.queens))
                    stack.push((neighbor.queens, current_depth + 1))

            self.queens = current_state[:]
            generated_nodes += stack.length() - start_length

            if self.conflicts() == 0:
                return current_state, current_time, current_memory, \
                    generated_nodes, max_nodes

        return None

    def BFS(self, initial_state, time_limit, memory_limit):
        start_time = time_ns()
        explored = set()
        queue = Queue()
        queue.push(initial_state)

        process = psutil.Process()
        max_nodes, generated_nodes = 0, 0

        while not queue.is_empty():
            node = queue.pop()
            current_time = (time_ns() - start_time) / 1e9
            current_memory = process.memory_info().rss / (1024 ** 2)
            max_nodes = max(max_nodes, queue.length())
            start_length = queue.length()

            explored.add(tuple(node))

            if current_time >= time_limit or current_memory >= memory_limit:
                return None

            for neighbor in self.neighbors():
                if neighbor not in explored:
                    queue.push(neighbor.queens)

            self.queens = node[:]
            generated_nodes += queue.length() - start_length

            if self.conflicts() == 0:
                return node, current_time, current_memory, \
                    generated_nodes, max_nodes

        return None

    def checking(self, queens, i, j):
        position_i = queens[i]
        position_j = queens[j]
        start = position_j

        for k in range(j - 1, i, -1):
            step = (position_i - position_j) / (j - i)
            temp_position = start + step

            if queens[k] == temp_position:
                self.checking(queens, i, k)
                return False

            start = temp_position

        return True

    def heuristic(self, queens=None):
        if queens is None: queens = self.queens[:]

        counter = 0

        for i in range(self.N - 1):
            row_i = queens[i]

            for j in range(i + 1, self.N):
                row_j = queens[j]

                if abs(row_i - row_j) == j - i and self.checking(queens, i, j):
                    counter += 1

        return counter

    def astar(self, initial_state, heuristic, time_limit, memory_limit):
        Node = namedtuple('Node', 'state parent cost')

        start_time = time_ns()
        process = psutil.Process()
        frontier = PriorityQueue()
        explored = dict()

        node = Node(self, None, 0)
        frontier.push(node, heuristic(queens=initial_state.queens))
        explored[initial_state] = node
        max_stored, generated_nodes, path_cost = 0, 0, 0

        while not frontier.is_empty():
            node = frontier.pop()
            current_time = (time_ns() - start_time) / 1e9
            current_memory = process.memory_info().rss / (1024 ** 2)
            max_stored = max(max_stored, len(explored))

            if current_time >= time_limit or current_memory >= memory_limit:
                return None

            for neighbor, step_cost in self.random_neighbor(queens=node.state.queens):
                generated_nodes += 1
                path_cost += node.cost + step_cost

                if neighbor not in explored or path_cost < explored[neighbor].cost:
                    child_node = Node(neighbor, node, path_cost)
                    explored[neighbor] = child_node
                    frontier.push(child_node, path_cost + heuristic(queens=neighbor.queens))

            self.queens = node.state.queens[:]
            generated_nodes += max_stored

            if self.conflicts(queens=node.state.queens) == 0:
                return node.state.queens, current_time, current_memory, \
                    generated_nodes, max_stored

        return None

    def recursive_rbfs(self, node, f_limit, path_cost, Node, heuristic, start_time, process,
                       generated_nodes, max_stored, time_limit, memory_limit):
        successors = []

        current_time = (time_ns() - start_time) / 1e9
        current_memory = process.memory_info().rss / (1024 ** 2)

        if current_time >= time_limit or current_memory >= memory_limit:
            return None, current_time, current_memory, \
                generated_nodes, max_stored

        if self.conflicts() == 0:
            return node.state.queens, current_time, current_memory, \
                generated_nodes, max_stored

        for neighbor in self.neighbors(queens=node.state.queens):
            generated_nodes += 1

            child_node = Node(neighbor, node, path_cost)
            successors.append((child_node, max(path_cost + heuristic(queens=neighbor.queens), f_limit)))

        if not successors:
            return None, current_time, current_memory, \
                generated_nodes, max_stored

        while len(successors):
            successors.sort(key=lambda x: x[1])
            best_node, best_f = successors[0]
            self.queens = best_node.state.queens[:]

            if best_f > f_limit:
                return self.queens, current_time, current_memory, \
                    generated_nodes, max_stored

            alternative_f = successors[1][1]
            result, current_time, current_memory, generated_nodes, max_stored = \
                self.recursive_rbfs(best_node, min(f_limit, alternative_f), path_cost + 1,
                                    Node, heuristic, start_time, process, generated_nodes, max_stored,
                                    time_limit, memory_limit)

            max_stored = max(max_stored, len(successors))

            if result is not None:
                return result, current_time, current_memory, \
                    generated_nodes, max_stored

        return None, current_time, current_memory, \
            generated_nodes, max_stored

    def rbfs(self, initial_state, heuristic, time_limit, memory_limit):
        Node = namedtuple('Node', 'state parent cost')

        start_time = time_ns()
        process = psutil.Process()

        node = Node(initial_state, None, 0)

        result, current_time, current_memory, generated_nodes, max_stored = \
            self.recursive_rbfs(node, self.heuristic(), 0, Node, heuristic,
                                start_time, process, 0, 0, time_limit, memory_limit)

        return result, current_time, current_memory, generated_nodes, max_stored

    @staticmethod
    def print_results(data, column=''):
        print("| {:<10} | {:<12} | {:<10} | {:<12} | {:<11} | {:<12} | {:<10} | {:<10} |"
        .format(
            'Iteration',
            'Initial state',
            'Algorithm',
            'Goal state',
            'Time',
            'Memory',
            'Generated',
            'Max. stored'
        ))

        total_time, avg_time, \
            total_generated, avg_generated = 0, 0, 0, 0
        valid_experiments = 0

        for i in data:
            temp_list = []

            for j in i:
                temp_list.extend(j) if isinstance(j, tuple) and j is not None \
                    else temp_list.append(j) if j is not None else temp_list.extend(['-----' for _ in range(5)])

            temp_list = ["".join(str(number) for number in elem)
                         if isinstance(elem, list) else elem for elem in temp_list]

            valid_experiments += 1 if i[len(i) - 1] is not None else 0

            try:
                total_time += temp_list[3] if type(temp_list[3] != str) else 0
                total_generated += temp_list[5] if type(temp_list[5] != str) else 0
            except TypeError:
                pass

            print("| {:<10} | {:<13} | {:<10} | {:<12} | {:<11} | {:<12} | {:<10} | {:<11} |"
                  .format(data.index(i) + 1, *temp_list))

        avg_time = total_time / len(data)
        avg_generated = total_generated / len(data)

        print("\nPath found times:", valid_experiments)
        print("Total time:", total_time)
        print("Average time:", avg_time)
        print("Total generated:", total_generated)
        # print("Average generated:", avg_generated)


if __name__ == '__main__':
    experiments_result = []

    for i in range(50):
        state = NQueensState(8)
        # print("Initial state:", state.queens)
        # print("Conflicts:", state.conflicts())

        state.plot()

        # random_depth = random.randint(2, 20)
        # print("Random depth:", random_depth)
        # experiments_result.append((state.queens, 'LDFS', state.LDFS(state.queens, random_depth, 10, 1024)))

        # random_depth = random.randint(2, 20)
        # print("Random depth:", random_depth)
        # experiments_result.append((state.queens, 'BFS', state.BFS(state.queens, 10, 1024)))

        # heuristic = state.heuristic()
        # print("Heuristic:", heuristic)
        # experiments_result.append((state.queens, 'A*', state.astar(state, state.heuristic, 10, 1024)))

        heuristic = state.heuristic()
        print("Heuristic:", heuristic)
        experiments_result.append((state.queens, 'RBFS', state.rbfs(state, state.heuristic, 10, 1024)))

        state.plot()

    # NQueensState.print_results(experiments_result, column='Depth')
    # NQueensState.print_results(experiments_result, column='Heuristic')
    NQueensState.print_results(experiments_result, column='Conflicts')
