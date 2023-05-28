from world import GridWorld
from agent import RandomAgent

world = GridWorld(10, 10, 3, 2)
agent = RandomAgent(world)

world.print_world()
# print(world.current_grid == world.last_grid)
# print("Start coords:", agent.start_location)

agent.value_iteration(10)
world.print_world()

# agent.find_solution()
