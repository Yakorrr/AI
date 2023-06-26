import math
import random
from collections import namedtuple
from time import time_ns

import matplotlib.pyplot as plt
import psutil
from tabulate import tabulate

from queue import Queue


class NPuzzleState:

    def __init__(self, N, tiles=None):
        self.N = N
        self.tiles = tuple(tiles[:] if tiles else self.random_state())
        self.grid_size = int(math.sqrt(N + 1))

    def __hash__(self):
        return hash(self.tiles)

    def __eq__(self, other):
        if self is other: return True
        if other is None: return False
        if not isinstance(other, NPuzzleState): return False

        return self.tiles == other.tiles

    def __str__(self):
        result = ''

        for i in range(len(self.tiles)):
            result += f' {self.tiles[i]:2d} ' if self.tiles[i] != 0 else '    '

            if i % self.grid_size == self.grid_size - 1 and i < self.N:
                result += '\n'

        return result

    def __repr__(self):
        return f'NPuzzleState(N={self.N}, tiles={self.tiles})'

    def random_state(self):
        puzzle = list(range(0, self.N + 1))
        random.shuffle(puzzle)

        return puzzle

    def successors(self):
        blank_idx = self.tiles.index(0)
        successors = []

        # left
        if blank_idx % self.grid_size > 0:
            tiles = list(self.tiles)
            tiles[blank_idx], tiles[blank_idx - 1] = tiles[blank_idx - 1], tiles[blank_idx]
            successor = NPuzzleState(self.N, tiles=tiles)
            successors.append((successor, 'Left', 1))

        # up
        if blank_idx >= self.grid_size:
            tiles = list(self.tiles)
            tiles[blank_idx], tiles[blank_idx - self.grid_size] = tiles[blank_idx - self.grid_size], tiles[blank_idx]
            successor = NPuzzleState(self.N, tiles=tiles)
            successors.append((successor, 'Up', 1))

        # right
        if blank_idx % self.grid_size < self.grid_size - 1:
            tiles = list(self.tiles)
            tiles[blank_idx], tiles[blank_idx + 1] = tiles[blank_idx + 1], tiles[blank_idx]
            successor = NPuzzleState(self.N, tiles=tiles)
            successors.append((successor, 'Right', 1))

        # down
        if blank_idx + self.grid_size < len(self.tiles):
            tiles = list(self.tiles)
            tiles[blank_idx], tiles[blank_idx + self.grid_size] = tiles[blank_idx + self.grid_size], tiles[blank_idx]
            successor = NPuzzleState(self.N, tiles=tiles)
            successors.append((successor, 'Down', 1))

        return successors

    def is_goal(self, goal_state):
        return self == goal_state

    def plot(self, ax=None, title="Start state", fs=18):
        if ax is None:
            _, ax = plt.subplots(1)

        gs = self.grid_size

        # draw border
        border = plt.Rectangle((0, 0), gs, gs, ec='k', fc='w', lw=3)
        ax.add_patch(border)

        # draw tiles
        for i, tile in enumerate(self.tiles):
            if tile == 0: continue

            col = self.grid_size - 1 - i // self.grid_size
            row = i % self.grid_size
            cell = plt.Rectangle((row, col), 1, 1, fc='darkslateblue', ec='k', lw=3, alpha=0.4)
            ax.add_patch(cell)
            tileSq = plt.Rectangle((row + 0.15, col + 0.15), 0.7, 0.7, fc='darkslateblue', ec='k', lw=1, alpha=0.8)
            ax.add_patch(tileSq)
            ax.text(row + 0.5, col + 0.5, f"{tile}", color='w', fontsize=fs, va='center', ha='center')

        ax.axis('square')
        ax.axis('off')
        ax.set_title(title, fontsize=fs)

        plt.show()

    @staticmethod
    def manhattan_distance(tile, state1, state2):
        i = state1.tiles.index(tile)
        j = state2.tiles.index(tile)

        gs = state1.grid_size

        row_i, col_i = i // gs, i % gs
        row_j, col_j = j // gs, j % gs

        return abs(row_i - row_j) + abs(col_i - col_j)

    def BFS(self, start_state, goal_state, time_limit, memory_limit):
        Node = namedtuple('Node', 'state parent action cost')

        explored = set()
        frontier = Queue()
        node = Node(start_state, None, None, 0)

        start_time = time_ns()
        process = psutil.Process()
        current_time, current_memory = 0, 0
        num_generated, max_nodes = 0, 0

        frontier.push(node)
        explored.add(start_state)

        while not frontier.is_empty():
            node = frontier.pop()
            current_time = (time_ns() - start_time) / 1e9
            current_memory = process.memory_info().rss / (1024 ** 2)
            max_nodes = max(max_nodes, frontier.length())

            if current_time >= time_limit or current_memory >= memory_limit:
                return None, current_time, current_memory, \
                    num_generated, max_nodes

            for successor, action, step_cost in node.state.successors():
                num_generated += 1

                if successor not in explored:
                    explored.add(successor)
                    frontier.push(Node(successor, node, action, node.cost + step_cost))

            self.tiles = tuple(node.state.tiles[:])

            if node.state == goal_state:
                return node.state.tiles, current_time, current_memory, \
                    num_generated, max_nodes

        return None, current_time, current_memory, \
            num_generated, max_nodes

    @staticmethod
    def print_results(data):
        rows = []

        for experiment in data:
            temp_row = []

            for elem in experiment:
                temp_row.extend(elem) if isinstance(elem, tuple) and any(
                    isinstance(el, tuple) for el in elem) else temp_row.extend(
                    ['-----' for _ in range(len(elem))]) if isinstance(
                    elem, tuple) and None in elem else temp_row.append(
                    elem) if isinstance(elem, tuple) else temp_row.append(elem)

            temp_row = ["".join(str(number) for number in elem)
                        if isinstance(elem, tuple) else elem for elem in temp_row]

            rows.append((data.index(experiment) + 1, *temp_row))

        print(tabulate(rows, headers=['Iteration', 'Initial state', 'Algorithm', 'Goal state',
                                      'Time', 'Memory', 'Generated', 'Max. stored'],
                       tablefmt='rounded_grid', stralign='center', numalign='center'))


def experiment_series(number):
    goal_state_tiles = [
        1, 2, 3,
        4, 5, 6,
        7, 8, 0]

    experiment_result = []
    goal_state = NPuzzleState(8, tiles=goal_state_tiles)

    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print(" GOAL STATE: ")
    print(goal_state)
    print("=======================================")

    for i in range(number):
        state = NPuzzleState(8)
        state.plot()

        print("%s. Start state:" % (i + 1))
        print(state)
        print("=======================================")

        experiment_result.append((state.tiles, "BFS", state.BFS(state, goal_state, 10, 1024)))

        state.plot(title="Goal state")

    NPuzzleState.print_results(experiment_result)


if __name__ == '__main__':
    experiment_numbers = int(input("Enter number of experiments: "))

    experiment_series(experiment_numbers)
