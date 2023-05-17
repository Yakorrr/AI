import random


class GridWorld:
    def __init__(self, width, height, bombs):
        self.width = width
        self.height = height
        self.grid = self.fill_grid()
        self.bombs_coordinates = self.place_bombs(bombs)
        self.room_width = self.room_height = 5
        self.corridors = self.place_corridors()
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

    def place_bombs(self, amount):
        def coordinates():
            return random.randint(0, self.width - 1), random.randint(0, self.height - 1)

        bombs_coords = []
        x, y = coordinates()

        for i in range(amount):
            while self.grid[x][y] == -50:
                x, y = coordinates()

            bombs_coords.append((x, y))
            self.grid[x][y] = -50

        return bombs_coords

    def place_corridors(self):
        corridors_coords = []
        current_width, current_height = self.room_width - 1, self.room_height - 1

        for i in range(0, self.width, self.room_width):
            print("Before L:", i, current_height)

            for j in range(0, self.height, self.room_height):
                print("Before H:", j, current_width)

                # if j < self.width - 1:
                temp_side = random.randint(j, current_width)
                corridors_coords.append((temp_side, j))
                self.grid[temp_side][j] = "||||"

                # if i < self.height - 1:
                temp_bottom = random.randint(i, current_height)
                corridors_coords.append((i, temp_bottom))
                self.grid[i][temp_bottom] = "||||"

                current_width += self.room_width

            current_height += self.room_height
            current_width = self.room_width - 1

        print(corridors_coords)

        return corridors_coords


world = GridWorld(25, 25, 10)
world.print_world()
