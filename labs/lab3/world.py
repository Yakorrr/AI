import random


class GridWorld:
    def __init__(self, width, height, bombs):
        self.width = width
        self.height = height
        self.room_width = self.room_height = 5

        self.grid = self.fill_grid()
        self.corridors = self.place_corridors()
        self.walls = self.place_walls()
        # self.bombs_coordinates = self.place_bombs(bombs)

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
        return random.randint(0, self.width - 1), random.randint(0, self.height - 1)

    def place_bombs(self, amount):
        bombs_coords = []

        for i in range(amount):
            x, y = self.random_coordinates()

            while self.grid[x][y] == -50 and (x, y) in self.corridors:
                x, y = self.random_coordinates()
                # print("X: %s, y: %s" % (x, y))

            bombs_coords.append((x, y))
            print("X: %s, y: %s" % (x, y))
            self.grid[x][y] = -50

        return bombs_coords

    def place_corridors(self):
        corridors_coords = []
        current_width, current_height = self.room_width - 1, self.room_height - 1

        for i in range(0, self.width, self.room_width):
            for j in range(0, self.height, self.room_height):
                if current_width < self.width - 1:
                    temp_side = random.randint(i, min(current_height - 1, self.height - 1))
                    corridors_coords.append((temp_side, current_width))
                    self.grid[temp_side][min(current_width, self.width - 1)] = "||||"

                if current_height < self.height - 1:
                    temp_bottom = random.randint(j, min(current_width - 1, self.width - 1))
                    corridors_coords.append((current_height, temp_bottom))
                    self.grid[min(current_height, self.height - 1)][temp_bottom] = "||||"

                current_width += self.room_width

            current_height += self.room_height
            current_width = self.room_width - 1

        return corridors_coords

    def place_walls(self):
        side, bottom = self.room_height, self.room_height

        for i in range(0, self.width, self.room_width):
            for j in range(0, self.height, self.room_height):
                print("Side:", j, side - 1)
                print("Bottom:", bottom - 1, i)

                for vertical in range(side - self.room_height, min(side, self.height - 1)):
                    if (vertical, bottom - 1) not in self.corridors:
                        self.grid[vertical][bottom - 1] = "####"

                for horizontal in range(bottom - self.room_width, min(bottom, self.width - 1)):
                    if (side - 1, horizontal) not in self.corridors:
                        self.grid[side - 1][horizontal] = "####"

                bottom += self.room_width

            side += self.room_height
            bottom = self.room_width

        # for i in self.corridors:
        #     x, y = i[0], i[1]
        #     print(x, y)
        #
        #     temp_x, temp_y = x % (self.room_height - 1), y % (self.room_width - 1)
        #
        #     if temp_x == 0 and temp_y != 0:
        #         for j in range(bottom, self.height):
        #             if (x, j) in self.corridors:
        #                 continue
        #             else:
        #                 self.grid[x][j] = "####"
        #
        #         bottom = x
        #
        #     if temp_y == 0 and temp_x != 0:
        #         for j in range(0, self.width):
        #             if (j, y) in self.corridors:
        #                 continue
        #             else:
        #                 self.grid[j][y] = "####"
        #
        #         side = y


world = GridWorld(10, 10, 10)
world.print_world()
