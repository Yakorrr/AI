import queue
from priority_queue import PriorityQueue
from world import Objects


class RandomAgent:
    def __init__(self, world):
        self.world = world

        self.temp_states = PriorityQueue()
        self.way = queue.Queue()

        self.slip_chance = 0.2
        self.probability = 1.0 - self.slip_chance
        self.penalty = 1.0
        self.reward = -1.0

    def value_iteration(self, iterations):
        for current_iteration in range(iterations):
            for i in range(0, self.world.width, 1):
                for j in range(0, self.world.height, 1):
                    if self.world.grid[i][j] == Objects.BOMB or self.world.grid[i][j] == Objects.GOAL or \
                            self.world.grid[i][j] == Objects.OBSTACLE:
                        self.world.current_grid[i][j] = self.world.last_grid[i][j]
                        continue

                    if i > 0:
                        self.temp_states.push((self.world.last_grid[i - 1][j], i, j))
                    elif i == 0 or self.world.grid[i - 1][j] == Objects.BOMB \
                            or self.world.grid[i - 1][j] == Objects.OBSTACLE:
                        self.temp_states.push((self.world.last_grid[i][j], i, j))

                    if j > 0:
                        self.temp_states.push((self.world.last_grid[i][j - 1], i, j))
                    elif j == 0 or self.world.grid[i][j - 1] == Objects.BOMB \
                            or self.world.grid[i][j - 1] == Objects.OBSTACLE:
                        self.temp_states.push((self.world.last_grid[i][j], i, j))

                    if i < self.world.width - 1:
                        self.temp_states.push((self.world.last_grid[i + 1][j], i, j))
                    elif i == self.world.width - 1 or self.world.grid[i + 1][j] == Objects.BOMB \
                            or self.world.grid[i + 1][j] == Objects.OBSTACLE:
                        self.temp_states.push((self.world.last_grid[i][j], i, j))

                    if j < self.world.height - 1:
                        self.temp_states.push((self.world.last_grid[i][j + 1], i, j))
                    elif j == self.world.height - 1 or self.world.grid[i][j + 1] == Objects.BOMB \
                            or self.world.grid[i][j + 1] == Objects.OBSTACLE:
                        self.temp_states.push((self.world.last_grid[i][j], i, j))

                    self.world.current_grid[i][j] = self.probability * (
                            self.reward + self.penalty * self.temp_states.pop()[0])

                    slipped = self.temp_states.length()

                    for _ in range(slipped):
                        self.world.current_grid[i][j] += (1 - self.probability) / slipped * (
                                self.reward + self.penalty * (self.temp_states.pop()[0]))

            self.world.last_grid = self.world.current_grid[:]

    # def find_solution(self):
    #     x, y = self.start_location[0], self.start_location[1]
    #     self.way.append((self.world.grid[x][y], x, y))
    #
    #     while self.world.grid[x][y] \
    #             not in self.world.rewards_coordinates:
    #         if x > 0: self.temp_states.append((self.world.grid[x - 1][y], x - 1, y))
    #         if y > 0: self.temp_states.append((self.world.grid[x][y - 1], x, y - 1))
    #
    #         if x < self.world.width - 1:
    #             self.temp_states.append((self.world.grid[x + 1][y], x + 1, y))
    #         if y < self.world.height - 1:
    #             self.temp_states.append((self.world.grid[x][y + 1], x, y + 1))
    #
    #         current = self.temp_states.pop()
    #         x, y = current[1], current[2]
    #         self.way.append(current)
    #
    #         while self.temp_states: self.temp_states.pop()
    #
    #     while self.way:
    #         current = self.way.pop(0)
    #         print("{:>8} {:>8} {:>8}".format(current[0], current[1], current[2]))
