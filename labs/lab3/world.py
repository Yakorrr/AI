import random
from enum import Enum


class Objects(Enum):
    NOTHING = 0.0
    GOAL = 100.0
    BOMB = -50.0
    OBSTACLE = -100


class GridWorld:
    def __init__(self, width, height, bombs, rewards):
        self.width = width
        self.height = height
        self.room_width = self.room_height = 5

        self.grid, self.current_grid, self.last_grid = self.fill_grids()

        self.corridors = self.place_corridors()
        self.place_walls()
        self.bombs_coordinates = self.place_item(bombs, Objects.BOMB)
        self.rewards_coordinates = self.place_item(rewards, Objects.GOAL)

    def __str__(self):
        return f'GridWorld {self.width}x{self.height}'

    def print_world(self):
        for row in self.current_grid:
            for elem in row:
                if elem == Objects.OBSTACLE.value:
                    print("{:>10}".format("####"), end='')
                else:
                    print("{:>10.2f}".format(elem), end='')

            print()

        print()

    def fill_grids(self):
        return [[Objects.NOTHING for _ in range(self.width)] for _ in range(self.height)], \
               [[Objects.NOTHING.value for _ in range(self.width)] for _ in range(self.height)], \
               [[Objects.NOTHING.value for _ in range(self.width)] for _ in range(self.height)]

    def random_coordinates(self):
        x, y = random.randint(0, self.width - 1), random.randint(0, self.height - 1)

        while self.grid[x][y] != Objects.NOTHING or self.grid[x][y] == Objects.OBSTACLE:
            x, y = random.randint(0, self.width - 1), random.randint(0, self.height - 1)

        return x, y

    def place_corridors(self):
        corridors_coords = []
        current_width, current_height = self.room_width - 1, self.room_height - 1

        for i in range(0, self.width, self.room_width):
            for j in range(0, self.height, self.room_height):
                if current_width < self.width - 1:
                    temp_side = random.randint(i, min(current_height - 1, self.height - 1))
                    corridors_coords.append((temp_side, current_width))

                if current_height < self.height - 1:
                    temp_bottom = random.randint(j, min(current_width - 1, self.width - 1))
                    corridors_coords.append((current_height, temp_bottom))

                current_width += self.room_width

            current_height += self.room_height
            current_width = self.room_width - 1

        return corridors_coords

    def place_walls(self):
        side, bottom = self.room_height, self.room_height

        while bottom < self.width:
            for horizontal in range(0, self.width):
                if bottom < self.width and (bottom - 1, horizontal) not in self.corridors:
                    self.grid[min(bottom - 1, self.width - 1)][horizontal] = Objects.OBSTACLE
                    self.current_grid[min(bottom - 1, self.width - 1)][horizontal] = Objects.OBSTACLE.value
                    self.last_grid[min(bottom - 1, self.width - 1)][horizontal] = Objects.OBSTACLE.value

            bottom += self.room_width

        while side < self.height:
            for vertical in range(0, self.height):
                if (vertical, side - 1) not in self.corridors:
                    self.grid[vertical][min(side - 1, self.height - 1)] = Objects.OBSTACLE
                    self.current_grid[vertical][min(side - 1, self.height - 1)] = Objects.OBSTACLE.value
                    self.last_grid[vertical][min(side - 1, self.height - 1)] = Objects.OBSTACLE.value

            side += self.room_height

        return None

    def place_item(self, amount, item):
        coords = []

        for _ in range(amount):
            x, y = self.random_coordinates()

            while (x, y) in self.corridors:
                x, y = self.random_coordinates()

            coords.append((x, y))
            self.grid[x][y] = item
            self.current_grid[x][y] = item.value
            self.last_grid[x][y] = item.value

        return coords
