# Moods (State Space)
MOODS = [
    "happy",
    "sad",
    "energetic",
    "calm",
    "angry",
    "romantic",
    "tired", 
    "depressed",
    "anxious",
    "party" 
]

# Music Categories (Action Space) - Optimized for YouTube Search Queries
MUSIC_Categories_List = [
    "Sad Bollywood",
    "Sad Punjabi",
    "Sad English Acoustic",
    "Happy Bollywood",
    "Upbeat Pop",
    "Energetic Rock",
    "Gym Phonk",
    "Punjabi Party",
    "Romantic Bollywood",
    "Romantic English",
    "Lo-Fi Study",
    "Jazz Instrumental",
    "Heavy Metal",
    "EDM Dance"
]

# Mapping Indices
MOOD_TO_INDEX = {mood: i for i, mood in enumerate(MOODS)}
INDEX_TO_MOOD = {i: mood for i, mood in enumerate(MOODS)}

CATEGORY_TO_INDEX = {cat: i for i, cat in enumerate(MUSIC_Categories_List)}
INDEX_TO_CATEGORY = {i: cat for i, cat in enumerate(MUSIC_Categories_List)}

# RL Hyperparameters
ALPHA = 0.1
GAMMA = 0.9
EPSILON = 0.2
