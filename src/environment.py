import random
import numpy as np
from src.config import MOODS, MOOD_TO_INDEX

class MoodEnvironment:
    def __init__(self):
        # Start with a random mood or a neutral one
        self.current_mood_index = random.randint(0, len(MOODS) - 1)
        self.setup_transitions()
    
    def setup_transitions(self):
        # Simple logical transition rules
        # Format: (CurrentMood, ActionCategory) -> [Probable Next Moods]
        # logic: 
        #   If Sad + Happy Song -> 60% Happy, 20% Energetic, 20% Sad (Didn't work)
        #   If Sad + Sad Song -> 50% Calm (Catharsis), 30% Sad, 20% Depressed
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
        # We'll use a simplified heuristic function instead of a full NxM density matrix for readability
        pass

    def set_mood(self, mood_name):
        if mood_name in MOOD_TO_INDEX:
            self.current_mood_index = MOOD_TO_INDEX[mood_name]

    def get_current_mood(self):
        return MOODS[self.current_mood_index]

    def get_reward(self, feedback, recommended_category):
        if feedback == "Super Like":
            return 20
        elif feedback == "add_to_playlist": # Explicit high reward
            return 20
        elif feedback == "Like":
            return 10
        elif feedback == "completed":
            return 5
        elif feedback == "skip_mid":
            return -2
        elif feedback == "skip_immediate":
            return -10
        elif feedback == "Dislike":
            return -10 # Increased penalty as requested
        elif feedback == "Wrong Vibe":
            return -10
        return 0

    def step(self, action_index, feedback):
        """
        Transitions mood based on feedback and action.
        """
        reward = self.get_reward(feedback, action_index)
        
        current_mood = MOODS[self.current_mood_index]
        # We don't have action_category name passed directly, assumes main app logic handles mapping,
        # but usually env should know the action. 
        # For this simplified env, let's just use heuristic rules based on Feedback.
        
        next_mood_index = self.current_mood_index
        
        # LOGIC:
        # If user LIKES the song, it generally moves them towards a "Positive" or "Desired" state
        # relative to the song's energy.
        
        if feedback == "Like":
            if current_mood in ["sad", "depressed", "angry"]:
                # Improvement!
                possible_next = ["calm", "neutral", "happy"]
                # Weighted choice: mostly calm (realistic)
                next_mood_name = random.choices(possible_next, weights=[0.6, 0.3, 0.1], k=1)[0]
                if next_mood_name in MOOD_TO_INDEX:
                   next_mood_index = MOOD_TO_INDEX[next_mood_name]
            
            elif current_mood in ["calm", "neutral"]:
                 # Can go to happy or energetic
                 possible_next = ["happy", "energetic", "romantic"]
                 next_mood_name = random.choice(possible_next)
                 if next_mood_name in MOOD_TO_INDEX:
                    next_mood_index = MOOD_TO_INDEX[next_mood_name]
                    
            elif current_mood in ["happy", "energetic", "party"]:
                 # Sustain (High probability to stay same)
                 if random.random() < 0.2:
                     next_mood_index = MOOD_TO_INDEX["tired"] # Burnout?
        
        elif feedback == "Dislike":
            # Mood might worsen
            if current_mood == "happy":
                next_mood_index = MOOD_TO_INDEX["angry"]
            elif current_mood == "calm":
                next_mood_index = MOOD_TO_INDEX["anxious"]
            elif current_mood == "sad":
                 # As requested: Sad + Dislike -> Angry (Frustration)
                next_mood_index = MOOD_TO_INDEX["angry"]
            elif current_mood == "energetic":
                next_mood_index = MOOD_TO_INDEX["tired"] # Drained

        # 10% Random Noise (Life happens)
        if random.random() < 0.1:
            next_mood_index = random.randint(0, len(MOODS) - 1)
            
        self.current_mood_index = next_mood_index
        return self.current_mood_index, reward
