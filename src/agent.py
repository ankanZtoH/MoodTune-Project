import numpy as np
import random
from src.config import MOODS, MUSIC_Categories_List, ALPHA, GAMMA, EPSILON

class QLearningAgent:
    def __init__(self, n_states=len(MOODS), n_actions=len(MUSIC_Categories_List), alpha=ALPHA, gamma=GAMMA, epsilon=EPSILON):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        
        # Initialize Q-table with zeros
        self.q_table = np.zeros((n_states, n_actions))

    def choose_action(self, state_index):
        """
        Choose an action based on Epsilon-Greedy policy.
        """
        if random.uniform(0, 1) < self.epsilon:
            # Explore: choose a random action
            return random.randint(0, self.n_actions - 1)
        else:
            # Exploit: choose the action with max Q-value
            # If there are multiple max values, choose randomly among them
            max_value = np.max(self.q_table[state_index])
            actions_with_max_value = np.where(self.q_table[state_index] == max_value)[0]
            return np.random.choice(actions_with_max_value)

    def learn(self, state_index, action_index, reward, next_state_index):
        """
        Update the Q-table using the Q-learning update rule.
        Q(s,a) <- Q(s,a) + alpha * [reward + gamma * max(Q(s', a')) - Q(s,a)]
        Returns details for visualization.
        """
        old_value = self.q_table[state_index, action_index]
        next_max = np.max(self.q_table[next_state_index])
        
        td_target = reward + self.gamma * next_max
        td_error = td_target - old_value
        
        new_value = old_value + self.alpha * td_error
        self.q_table[state_index, action_index] = new_value
        
        return old_value, new_value, td_error
