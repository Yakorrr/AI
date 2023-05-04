import random

import numpy as np
import matplotlib.pyplot as plt


class GridWorld:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.grid = self.fill_grid()

    def __str__(self):
        matrix_str = ""

        for i in range(len(self.grid)):
            for j in range(len(self.grid[i])):
                matrix_str += str(self.grid[i][j]) + " " * 4

            matrix_str += "\n"

        return matrix_str[:-1]

    def fill_grid(self):
        return [[0 for _ in range(self.width)] for _ in range(self.height)]

    def print_world(self):
        for i in self.grid:
            print(i, end='\n')

    # self.bombs = self.place_bombs((width + height) / )

    # def place_bombs(self, amount):
    #     for i in range(amount):


world = GridWorld(10, 10)
print(world)
