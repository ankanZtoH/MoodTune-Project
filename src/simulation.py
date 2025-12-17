import random
from src.config import MOODS, MUSIC_Categories_List

class SimulatedUser:
    """
    A simulated user with defined preferences to train the agent offline.
    """
    def __init__(self):
        # Preferences: {Mood: [Liked Categories]}
        # Anything not in Liked is considered Disliked or Neutral
        self.preferences = {
            "happy": ["Upbeat Pop", "Party", "Happy Bollywood", "Punjabi Party", "EDM Dance"],
            "sad": ["Sad Bollywood", "Sad Punjabi", "Sad English Acoustic", "Lo-Fi Study"],
            "energetic": ["Energetic Rock", "Gym Phonk", "Hip-Hop", "Workout", "Upbeat Pop"], # Workout/HipHop map to categories if exist
            "calm": ["Lo-Fi Study", "Jazz Instrumental", "Classical", "Romantic English"],
            "angry": ["Heavy Metal", "Energetic Rock", "Rock"], # Catharsis
            "romantic": ["Romantic Bollywood", "Romantic English", "Jazz Instrumental"],
            "tired": ["Lo-Fi Study", "Jazz Instrumental", "Calm"],
            "depressed": ["Sad Bollywood", "Sad Punjabi", "Heavy Metal"], # Sometimes metal helps?
            "anxious": ["Lo-Fi Study", "Classical", "Jazz Instrumental"], # Calming
            "party": ["Punjabi Party", "EDM Dance", "Upbeat Pop", "Happy Bollywood"]
        }

    def get_feedback(self, mood, category):
        """
        Returns simulated feedback based on mood and category.
        """
        # 1. Check if category is explicitly liked for this mood
        liked_cats = self.preferences.get(mood, [])
        
        if category in liked_cats:
            # 80% chance of Like, 10% Super Like, 10% Add to Playlist
            r = random.random()
            if r < 0.8: return "Like"
            elif r < 0.9: return "Super Like"
            else: return "add_to_playlist"
            
        # 2. Check for severe mismatches (Dislike/Skip)
        # e.g., Sad mood + Party music = Hate it
        if mood in ["sad", "depressed"] and category in ["Party", "Punjabi Party", "EDM Dance", "Upbeat Pop"]:
            return "skip_immediate" # Hated it
            
        if mood in ["calm", "tired", "anxious"] and category in ["Heavy Metal", "Energetic Rock", "Gym Phonk"]:
            return "skip_immediate" # Too loud
            
        # 3. Default: Neutral / Mild Dislike / Skip Mid
        # If it's not a match, they probably skip it eventually
        return "skip_mid"
