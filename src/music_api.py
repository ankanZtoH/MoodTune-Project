from ytmusicapi import YTMusic
import random

# Initialize API (Unauthenticated is fine for search)
yt = YTMusic()

def search_songs_by_category(category, limit=5):
    """
    Searches YouTube Music for songs matching the category.
    Returns a list of structured song dicts.
    """
    search_query = f"Best {category} songs"
    
    try:
        # Search for songs
        results = yt.search(search_query, filter="songs", limit=limit)
        
        songs = []
        for track in results:
            song_data = {
                'title': track.get('title', 'Unknown Title'),
                'artist': ', '.join([artist['name'] for artist in track.get('artists', [])]),
                'genre': category, # Approximate
                'mood': category, # Approximate
                # Use the largest thumbnail
                'cover_url': track['thumbnails'][-1]['url'] if track.get('thumbnails') else 'https://via.placeholder.com/300',
                # Construct video link
                'link': f"https://music.youtube.com/watch?v={track['videoId']}"
            }
            songs.append(song_data)
            
        return songs
    except Exception as e:
        print(f"API Error: {e}")
        return []

def get_dynamic_recommendation(mood_category):
    """
    Fetches a random song from top results for the given mood/category.
    """
    songs = search_songs_by_category(mood_category, limit=10)
    if songs:
        return random.choice(songs)
    return None
