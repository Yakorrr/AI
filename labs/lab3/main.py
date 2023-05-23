from world import GridWorld
from agent import RandomAgent

world = GridWorld(10, 10, 5)
agent = RandomAgent(world, 1, 1)

world.print_world()
print("Start coords:", agent.current_location)
