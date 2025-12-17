from textblob import TextBlob
import random
import hashlib
import numpy as np
from PIL import Image
import io

# Try importing DeepFace, but don't crash if it's not installed yet or fails
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False

def detect_mood_text(text):
    """
    Detects mood from text using Keyword Heuristics + Sentiment Analysis.
    """
    if not text:
        return None
        
    text_lower = text.lower()
    
    # 1. Keyword Heuristics (Override Sentiment)
    keywords = {
        "birthday": "happy",
        "party": "party",
        "celebration": "party",
        "love": "romantic",
        "crush": "romantic",
        "date": "romantic",
        "breakup": "sad",
        "died": "sad",
        "lost": "sad",
        "exam": "anxious",
        "interview": "anxious",
        "nervous": "anxious",
        "gym": "energetic",
        "workout": "energetic",
        "run": "energetic",
        "sleep": "tired",
        "tired": "tired",
        "exhausted": "tired",
        "relax": "calm",
        "chill": "calm"
    }
    
    for word, mood in keywords.items():
        if word in text_lower:
            return mood

    # 2. Fallback to Sentiment Analysis
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    
    # Simple mapping
    if polarity > 0.5:
        return "happy"
    elif polarity > 0.1:
        return "energetic"
    elif polarity < -0.5:
        return "depressed"
    elif polarity < -0.1:
        return "sad"
    else:
        return "calm"

def get_deterministic_mock_mood(image_bytes):
    """
    Returns a consistent mood for the same image input using hashing.
    """
    moods = ["happy", "sad", "calm", "energetic", "angry", "neutral"]
    # Create a hash of the image bytes
    m = hashlib.md5()
    m.update(image_bytes)
    # Use the hash integer to select a mood index
    hash_int = int(m.hexdigest(), 16)
    return moods[hash_int % len(moods)]

def detect_mood_visual(image_buffer):
    """
    Visual Mood Detection using DeepFace (if available) or Deterministic Mock.
    """
    if not image_buffer:
        return None
        
    # Convert buffer to bytes for hashing/processing
    image_bytes = image_buffer.getvalue()
    
    if DEEPFACE_AVAILABLE:
        try:
            # DeepFace expects an image path or numpy array. 
            # We convert the buffer to a numpy array.
            image = Image.open(image_buffer)
            image_np = np.array(image)
            
            # Analyze
            predictions = DeepFace.analyze(image_np, actions=['emotion'], enforce_detection=False)
            
            # Result is a list of dicts (for each face). Take the first one.
            if predictions and isinstance(predictions, list):
                dominant_emotion = predictions[0]['dominant_emotion']
                return dominant_emotion # e.g., 'happy', 'sad', 'neutral'
            elif predictions and isinstance(predictions, dict):
                 return predictions['dominant_emotion']
                 
        except Exception as e:
            print(f"DeepFace error: {e}")
            # Fallback to mock if DeepFace crashes (e.g. face not found)
            return get_deterministic_mock_mood(image_bytes)
    
    # Fallback if no DeepFace
    return get_deterministic_mock_mood(image_bytes)
