from world import GridWorld
from agent import RandomAgent

if __name__ == '__main__':
    world = GridWorld(25, 25, 10, 10)
    agent = RandomAgent(world)

    world.print_world()

    agent.value_iteration(100)
    world.print_world()

    print("Start coordinates:", agent.start_location)
    agent.find_solution()
