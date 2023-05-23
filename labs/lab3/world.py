import random


class GridWorld:
    def __init__(self, width, height, bombs):
        self.width = width
        self.height = height
        self.room_width = self.room_height = 5

        self.grid = self.fill_grid()
        self.corridors = self.place_corridors()
        self.place_walls()
        self.bombs_coordinates = self.place_bombs(bombs)

        self.slip_chance = 0.1
        self.probability = 1 - self.slip_chance
        self.penalty = 0.8
        self.reward = -1

    def __str__(self):
        return f'GridWorld {self.width}x{self.height}'

    def print_world(self):
        for row in self.grid:
            for elem in row:
                print("{:>10}".format(elem), end='')

            print()

    def fill_grid(self):
        return [[0 for _ in range(self.width)] for _ in range(self.height)]

    def random_coordinates(self):
        x, y = random.randint(0, self.width - 1), random.randint(0, self.height - 1)

        while self.grid[x][y] == -50 or self.grid[x][y] == "####":
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
                    self.grid[min(bottom - 1, self.width - 1)][horizontal] = "####"

            bottom += self.room_width

        while side < self.height:
            for vertical in range(0, self.height):
                if (vertical, side - 1) not in self.corridors:
                    self.grid[vertical][min(side - 1, self.height - 1)] = "####"

            side += self.room_height

        return None

    def place_bombs(self, amount):
        bombs_coords = []

        for i in range(amount):
            x, y = self.random_coordinates()

            while (x, y) in self.corridors:
                x, y = self.random_coordinates()

            bombs_coords.append((x, y))
            self.grid[x][y] = -50

        return bombs_coords

    def count_states(self):
        states = 0

        for i in self.grid:
            for j in i:
                if j != '####': states += 1

        return states
