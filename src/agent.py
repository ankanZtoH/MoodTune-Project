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
        self.history = [] # Training History (Rewards)
        self.mood_history = [] # History of visited states (moods)
        self.action_history = [] # History of actions taken (categories)
        
        # Try Loading, else Pre-train
        if not self.load_model():
             self.initialize_knowledge()

    def save_model(self, filename="brain.npz"):
        try:
            np.savez(filename, 
                     q_table=self.q_table, 
                     history=self.history,
                     mood_history=self.mood_history,
                     action_history=self.action_history
            )
            return True
        except Exception as e:
            print(f"Error saving model: {e}")
            return False

    def load_model(self, filename="brain.npz"):
        try:
            data = np.load(filename)
            self.q_table = data['q_table']
            
            # Safe loading for history fields (backward compatibility)
            if 'history' in data: self.history = list(data['history'])
            else: self.history = []
            
            if 'mood_history' in data: self.mood_history = list(data['mood_history'])
            else: self.mood_history = []
            
            if 'action_history' in data: self.action_history = list(data['action_history'])
            else: self.action_history = []
                
            print("Model loaded successfully.")
            return True
        except FileNotFoundError:
            return False
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def initialize_knowledge(self):
        """
        Injects common-sense mappings into Q-Table so the agent isn't dumb at start.
        """
        from src.config import MOOD_TO_INDEX, MUSIC_Categories_List                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
        
        # Helper to set Q-value safely
        def set_q(mood, category, value):
            if mood in MOOD_TO_INDEX and category in MUSIC_Categories_List:
                m_idx = MOOD_TO_INDEX[mood]
                c_idx = MUSIC_Categories_List.index(category)
                self.q_table[m_idx, c_idx] = value

        # --- HEURISTICS ---
        # Happy People usually like:
        set_q("happy", "Upbeat Pop", 5.0)
        set_q("happy", "Party", 4.0)
        set_q("happy", "Happy", 5.0)
        
        # Sad People usually like:
        set_q("sad", "Sad Bollywood", 5.0)
        set_q("sad", "Sad Punjabi", 4.0)
        set_q("sad", "Lo-Fi", 2.0)
        
        # Angry People usually like (Catharsis or Calming):
        set_q("angry", "Rock", 4.0)
        set_q("angry", "Metal", 3.0)
        set_q("angry", "Calm", 2.0) # To calm down
        
        # Calm People usually like:
        set_q("calm", "Lo-Fi", 5.0)
        set_q("calm", "Classical", 4.0)
        
        # Energetic People usually like:
        set_q("energetic", "Workout", 5.0)
        set_q("energetic", "Hip-Hop", 4.0)
        
        # Surprise (Wildcard): Slightly lower values to encourage exploration
        set_q("surprise", "Party", 2.0)

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

    def train_offline(self, episodes=200, progress_callback=None, delay=0.0):
        """
        Train the agent using a simulated user for N episodes.
        """
        import time
        from src.environment import MoodEnvironment
        from src.simulation import SimulatedUser
        from src.config import MOOD_TO_INDEX, INDEX_TO_CATEGORY, MOODS

        env = MoodEnvironment()
        sim_user = SimulatedUser()
        
        print(f"Starting offline training for {episodes} episodes...")
        
        for i in range(episodes):
            if delay > 0:
                time.sleep(delay)
                
            # 1. Start with random mood
            current_mood_idx = env.current_mood_index
            current_mood_str = MOODS[current_mood_idx]
            
            # 2. Agent chooses action
            action_idx = self.choose_action(current_mood_idx)
            action_category = INDEX_TO_CATEGORY[action_idx]
            
            # 3. Simulated User gives feedback
            feedback = sim_user.get_feedback(current_mood_str, action_category)
            
            # 4. Environment Step
            next_mood_idx, reward = env.step(action_idx, feedback)
            
            # 5. Learn
            self.learn(current_mood_idx, action_idx, reward, next_mood_idx)
            
            # --- TRACKING ---
            self.history.append(reward)
            self.mood_history.append(current_mood_str)
            self.action_history.append(action_category)
            
            # Update env for next step
            env.current_mood_index = next_mood_idx
            
            if progress_callback:
                progress_callback(i + 1, episodes)
            
        print("Offline training complete.")
        self.save_model() # Auto save after training
        return True
