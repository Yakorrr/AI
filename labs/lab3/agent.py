import numpy as np


class RandomAgent:
    def __init__(self, world, gamma, theta):
        self.world = world
        self.max_iterations = 1e3
        self.gamma = gamma
        self.theta = theta

        self.actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        self.current_location = world.random_coordinates()

        self.num_states = world.count_states()
        self.num_actions = self.agent_actions()

        self.values = np.zeros(self.num_states)
        self.policy = np.zeros((self.num_states, self.num_actions))

    def agent_actions(self):
        return len(self.actions)

    # def move(self, action):
    #     last_location = self.current_location
    #
    #     if action == 'UP':

    # def value_iteration(self, iterations):
    #     for current_iteration in range(iterations):
    #         for i in range(world.width):
    #             for j in range(world.height):

    # iteration = 0
    #
    # while True:
    #     delta = 0
    #     previous_values = np.copy(self.values)
    #
    #     for state in range(self.num_states):
    #         Q_value = []
    #         old_value = self.values[state]
    #
    #         for action in range(self.num_actions):
    #             next_states_rewards = []
    #
    #             for trans_prob, next_state, reward_prob, _ in self.state:
