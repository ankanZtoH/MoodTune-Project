import streamlit as st
import pandas as pd
import numpy as np
import random
import altair as alt # Moved to top level
from src.config import MOODS, MOOD_TO_INDEX, MUSIC_Categories_List, INDEX_TO_CATEGORY
from src.agent import QLearningAgent
from src.environment import MoodEnvironment
from src.mood_detection import detect_mood_text, detect_mood_visual
from src.music_api import get_dynamic_recommendation

# --- Page Config ---
st.set_page_config(page_title="MoodTunes", page_icon="🎵", layout="wide", initial_sidebar_state="expanded")

# --- Custom CSS (Updated for Wireframe) ---
st.markdown("""
<style>
    .stApp {
        background-color: #000000;
        color: white;
    }
    
    /* Card Container */
    .player-card {
        background-color: #121212;
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.6);
        margin-top: 50px;
    }
    
    /* Typography */
    .song-title {
        font-size: 1.8rem;
        font-weight: 700;
        color: white;
        margin-bottom: 0px;
    }
    .artist-name {
        font-size: 1.1rem;
        color: #b3b3b3;
        margin-bottom: 10px;
    }
    
    /* Progress Bar Simulation */
    .progress-track {
        background-color: #535353;
        height: 6px;
        border-radius: 3px;
        wi background-color: #121212;
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.6);
        margin-top: 20px;dth: 100%;
        margin: 15px 0;
        position: relative;
    }
    .progress-fill {
        background-color: #1DB954;
        height: 100%;
        width: 30%; /* Mock Progress */
        border-radius: 3px;
    }
    
    /* Button Layout */
    .stButton>button {
        font-weight: 600;
        border-radius: 8px;
        border: 1px solid #333;
        background-color: #2a2a2a;
        color: #ddd;
        transition: 0.2s;
        width: 100%;
    }
    .stButton>button:hover {
        border-color: #1DB954;
        color: #1DB954;
    }
    
    /* Video Player Styling */
    .stVideo {
        border-radius: 12px;
        overflow: hidden;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #050505;
        border-right: 1px solid #222;
    }
    
    /* Mood Detect Visuals */
    .mood-detected {
        font-size: 1.2rem;
        font-weight: bold;
        color: #1DB954; 
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

# --- Initialization ---
if 'agent' not in st.session_state:
    st.session_state.agent = QLearningAgent(len(MOODS), len(MUSIC_Categories_List))
if 'environment' not in st.session_state:
    st.session_state.environment = MoodEnvironment()

if 'current_song' not in st.session_state: st.session_state.current_song = None
if 'detected_mood' not in st.session_state: st.session_state.detected_mood = None
if 'last_action_idx' not in st.session_state: st.session_state.last_action_idx = None
if 'last_state_idx' not in st.session_state: st.session_state.last_state_idx = None
if 'next_state_idx' not in st.session_state: st.session_state.next_state_idx = None # Added for Internal Details
if 'last_reward' not in st.session_state: st.session_state.last_reward = 0
if 'last_image_buffer' not in st.session_state: st.session_state.last_image_buffer = None
if 'playlist' not in st.session_state: st.session_state.playlist = []
if 'show_video' not in st.session_state: st.session_state.show_video = False
if 'start_time' not in st.session_state: st.session_state.start_time = 0

if 'last_learning_details' not in st.session_state: st.session_state.last_learning_details = None

# --- Logic Helper ---
def handle_feedback_and_next(feedback):
    # Logic for Skip: Treat as Negative Feedback & Next Song
    # Removed the 'seeking' logic as per user request (Skip = Not Interested)

    next_mood_idx, reward = st.session_state.environment.step(st.session_state.last_action_idx, feedback)
    st.session_state.last_reward = reward
    st.session_state.agent.history.append(reward) # Track for graph
    
    if st.session_state.last_state_idx is not None and st.session_state.last_action_idx is not None:
        old_v, new_v, td_err = st.session_state.agent.learn(
            st.session_state.last_state_idx, 
            st.session_state.last_action_idx, 
            reward, 
            next_mood_idx
        )
        st.session_state.last_learning_details = {
            "old": old_v,
            "new": new_v,
            "td_error": td_err
        }
        
    # --- TRACKING (User Interaction) ---
    current_mood_str = st.session_state.detected_mood
    action_category = INDEX_TO_CATEGORY.get(st.session_state.last_action_idx, "Unknown")
    
    st.session_state.agent.mood_history.append(current_mood_str)
    st.session_state.agent.action_history.append(action_category)
    
    st.session_state.agent.save_model() # Auto-save for realistic persistence
    
    prev_mood = st.session_state.detected_mood
    next_mood_name = MOODS[next_mood_idx]
    
    # Update State Tracking for Display
    st.session_state.next_state_idx = next_mood_idx
    
    st.toast(f"Recorded: {feedback} (Reward: {reward})")
    play_music_for_mood(next_mood_name)
    st.rerun()

# Helper for feedback without skipping (Added for Playlist logic)
def handle_feedback_only(feedback):
    next_mood_idx, reward = st.session_state.environment.step(st.session_state.last_action_idx, feedback)
    st.session_state.last_reward = reward
    st.session_state.agent.history.append(reward)
    
    if st.session_state.last_state_idx is not None and st.session_state.last_action_idx is not None:
         # Learn but stay on same song
         old_v, new_v, td_err = st.session_state.agent.learn(
            st.session_state.last_state_idx, 
            st.session_state.last_action_idx, 
            reward, 
            MOOD_TO_INDEX.get(st.session_state.detected_mood, 0) # Fixed: Convert string to index 
         )
         st.session_state.last_learning_details = { "old": old_v, "new": new_v, "td_error": td_err }
         
         # --- TRACKING (User Interaction) ---
         current_mood_str = st.session_state.detected_mood
         action_category = INDEX_TO_CATEGORY.get(st.session_state.last_action_idx, "Unknown")
         
         st.session_state.agent.mood_history.append(current_mood_str)
         st.session_state.agent.action_history.append(action_category)
         
         st.session_state.agent.save_model() # Auto-save for realistic persistence
         
    st.toast(f"Added to Playlist! (Reward: {reward})")

def play_music_for_mood(mood):
    mood = mood.lower()
    st.session_state.environment.set_mood(mood)
    st.session_state.detected_mood = mood
    st.session_state.last_state_idx = MOOD_TO_INDEX.get(mood, 0)
    
    action_idx = st.session_state.agent.choose_action(st.session_state.last_state_idx)
    target_category = INDEX_TO_CATEGORY[action_idx]
    st.session_state.last_action_idx = action_idx
    
    with st.spinner(f"Curating '{target_category}'..."):
        song_data = get_dynamic_recommendation(target_category)
        
    if song_data:
        st.session_state.current_song = song_data
        st.session_state.start_time = 0 # Reset start time for new song
    else:
        st.error("API Error - Could not fetch song.")

# --- UI IMPLEMENTATION ---

# 1. Sidebar (Logo, Playlist, Internal Details, QTable)
with st.sidebar:
    # Logo Area
    st.header("MOODTUNES")
    st.caption("AI-Powered Music Player")
    st.divider()

    # My Playlist
    st.subheader("My Playlist")
    if st.session_state.playlist:
        for i, item in enumerate(st.session_state.playlist):
             if st.button(f"{i+1}. {item['title']}", key=f"pl_{i}"):
                 st.session_state.current_song = item
                 st.rerun()
        if st.button("Clear Playlist"):
            st.session_state.playlist = []
            st.rerun()
    else:
        st.write("o\no\no") # Wireframe style placeholder look
        st.caption("No songs saved yet.")

    st.divider()

    # Internal Details - UPDATED with RL Specifics
    st.subheader("Internal Details")
    
    # Getting values safely
    curr_s = st.session_state.detected_mood if st.session_state.detected_mood else "None"
    last_act = INDEX_TO_CATEGORY.get(st.session_state.last_action_idx, "None") if st.session_state.last_action_idx is not None else "None"
    next_s = MOODS[st.session_state.next_state_idx] if st.session_state.next_state_idx is not None else "None"
    
    st.markdown(f"""
    **State & Transition:**
    - **State ($s$):** `{curr_s}`
    - **Action ($a$):** `{last_act}`
    - **Reward ($r$):** `{st.session_state.last_reward}`
    - **Next State ($s'$):** `{next_s}`
    """)
    
    if st.session_state.last_learning_details:
        ld = st.session_state.last_learning_details
        st.markdown(f"""
        **Learning Update ($Q(s,a)$):**
        - Old Value: `{ld['old']:.3f}`
        - TD Error: `{ld['td_error']:.3f}`
        - **New Value:** `{ld['new']:.3f}`
        """)
    else:
        st.caption("No learning step recorded yet.")

    st.divider()
    
    # Qtable
    st.subheader("Qtable:")
    q_df = pd.DataFrame(st.session_state.agent.q_table, index=MOODS, columns=MUSIC_Categories_List)
    st.dataframe(q_df.style.background_gradient(cmap="viridis", axis=1), height=200)

    st.divider()
    
    # --- Training Controls & Metrics ---
    st.subheader("Training Manager")
    
    # Training Stats
    total_eps = len(st.session_state.agent.history)
    st.write(f"**🧠 Brain Maturity:** `{total_eps} Episodes`")
    
    # Controls
    tc1, tc2 = st.columns(2)
    with tc1:
        if st.button("Save Brain"):
            if st.session_state.agent.save_model():
                st.toast("Model Saved Successfully!")
            else:
                st.error("Save Failed")
        
        # Pre-train Button
        if st.button("Pre-train (200 Eps)"):
            p_bar = st.progress(0, text="Training Agent...")
            def update_progress(current, total):
                p_bar.progress(current / total, text=f"Training Episode {current}/{total}")
            
            with st.spinner("Training Agent..."):
                # Total time = 200 * 0.05 = 10 seconds
                st.session_state.agent.train_offline(200, progress_callback=update_progress, delay=0.05)
            
            p_bar.empty() # Remove progress bar
            st.toast("Training Complete! Graph Updated.")
            st.rerun()
    with tc2:
        if st.button("Reset Brain"):
             import os
             if os.path.exists("brain.npz"):
                 os.remove("brain.npz")
             st.session_state.agent.q_table = np.zeros((st.session_state.agent.n_states, st.session_state.agent.n_actions))
             st.session_state.agent.history = []
             st.session_state.agent.initialize_knowledge() # Re-apply heuristics
             st.toast("Brain Reset to Factory Settings")
             st.rerun()

    # Metrics Plot
    st.write("**Learning Curve (Episode vs Reward):**")
    if st.session_state.agent.history:
        
        # Prepare Data
        history = st.session_state.agent.history
        df = pd.DataFrame({
            "Episode": range(1, len(history) + 1),
            "Reward": history
        })
        # Calculate Moving Average (Window=10 or 10% of total)
        window = max(5, int(len(history) * 0.05))
        df["Trend"] = df["Reward"].rolling(window=window, min_periods=1).mean()
        
        # Create Chart
        base = alt.Chart(df).encode(x='Episode')
        
        # Raw Rewards (Scatter/Light Line)
        raw = base.mark_circle(opacity=0.3, color='gray').encode(
            y='Reward',
            tooltip=['Episode', 'Reward']
        )
        
        # Trend Line
        trend = base.mark_line(color='#1DB954', strokeWidth=3).encode(
            y=alt.Y('Trend', title='Reward'),
            tooltip=['Episode', 'Trend']
        )
        
        c = (raw + trend).properties(height=200).interactive()
        st.altair_chart(c, use_container_width=True)
        
    else:
        st.caption("Start giving feedback to see the learning curve!")

    st.divider()
    
    # --- New Analytics Dashboard ---
    with st.expander("📊 Analytics Dashboard", expanded=True):
        if st.session_state.agent.mood_history:
            # 1. Mood Distribution
            st.markdown("**Mood Distribution**")
            mood_counts = pd.Series(st.session_state.agent.mood_history).value_counts().reset_index()
            mood_counts.columns = ['Mood', 'Count']
            
            c_mood = alt.Chart(mood_counts).mark_arc(innerRadius=50).encode(
                theta=alt.Theta("Count", stack=True),
                color=alt.Color("Mood", scale=alt.Scale(scheme='category20b')),
                tooltip=["Mood", "Count"]
            ).properties(height=200)
            st.altair_chart(c_mood, use_container_width=True)
            
            # 2. Genre Distribution
            st.markdown("**Genre Distribution (Recommendations)**")
            action_counts = pd.Series(st.session_state.agent.action_history).value_counts().reset_index()
            action_counts.columns = ['Genre', 'Count']
            
            c_genre = alt.Chart(action_counts).mark_bar().encode(
                x=alt.X('Count'),
                y=alt.Y('Genre', sort='-x'),
                color=alt.Color('Genre', legend=None),
                tooltip=['Genre', 'Count']
            ).properties(height=200)
            st.altair_chart(c_genre, use_container_width=True)
        else:
            st.caption("No history data available for distributions.")

        # 3. Q-Table Heatmap
        st.markdown("**Mood-Resource Relationship (Q-Table Heatmap)**")
        
        # Transform Q-table to long format for Altair
        q_table = st.session_state.agent.q_table
        data_q = []
        for i, mood in enumerate(MOODS):
            for j, genre in enumerate(MUSIC_Categories_List):
                data_q.append({"Mood": mood, "Genre": genre, "Q-Value": round(q_table[i, j], 2)})
        
        df_q = pd.DataFrame(data_q)
        
        # Base Chart
        base = alt.Chart(df_q).encode(
            x='Genre:O',
            y='Mood:O'
        )
        
        # Heatmap Layer
        heatmap = base.mark_rect().encode(
            color=alt.Color('Q-Value:Q', scale=alt.Scale(scheme='plasma'), legend=alt.Legend(title="Q-Value")),
            tooltip=['Mood', 'Genre', 'Q-Value']
        )
        
        # Text Layer
        text = base.mark_text(baseline='middle').encode(
            text='Q-Value:Q',
            color=alt.value('white')  # Fixed text color for contrast on dark plasma
        )
        
        # Combine
        c_heat = (heatmap + text).properties(height=300)
        st.altair_chart(c_heat, use_container_width=True)


# Main Layout (Middle and Right Columns)
col_mid, col_right = st.columns([1, 1.3], gap="large")

# 2. Middle Column (Mood Detect Option, Space, Result)
with col_mid:
    st.subheader("Mood Detect Option")
    
    # Custom Tabs for Options
    detect_option = st.radio("Select Input Method", ["Visual", "Text", "Manual"], horizontal=True, label_visibility="collapsed")
    
    detected_now = None

    # Mood Detect Space (Dynamic based on selection)
    # Removing fixed height to prevent internal scrolling for Camera
    with st.container(border=True):
        if detect_option == "Visual":
            st.write("### Camera Input")
            # Using columns to constrain width, which effectively reduces height (keeping aspect ratio)
            # This makes the "box small" as requested
            vc1, vc2, vc3 = st.columns([0.1, 0.8, 0.1])
            with vc2:
                cam = st.camera_input("Scan Face", label_visibility="collapsed")
            
            if cam:
                current_bytes = cam.getvalue()
                if st.session_state.last_image_buffer != current_bytes:
                        st.session_state.last_image_buffer = current_bytes
                        d = detect_mood_visual(cam)
                        if d: 
                            st.session_state.detected_mood = d
                            detected_now = d
        
        elif detect_option == "Text":
            st.write("### Text Input")
            txt = st.text_area("How do you feel?", height=150)
            if st.button("Analyze Text"):
                d = detect_mood_text(txt)
                if d: 
                    st.session_state.detected_mood = d
                    detected_now = d
        
        elif detect_option == "Manual":
            st.write("### Manual Selection")
            sel = st.selectbox("Choose Mood", MOODS)
            if st.button("Set Mood"):
                    st.session_state.detected_mood = sel
                    detected_now = sel

    # Take Photo/Remove Photo (Simulated)
    if st.button("Reset / Clear", use_container_width=True):
        st.session_state.detected_mood = None
        st.rerun()

    st.divider()
    
    # Detected Mood Section
    cur_mood = st.session_state.detected_mood
    if cur_mood:
        st.markdown(f"**Detected Mood:** {cur_mood.title()}")
    else:
        st.markdown("**Detected Mood:** ...")
        st.caption("exp. Happy") # Only show example at beginning

    st.markdown("---")
    
    # Play Recommend Song
    if st.button("Play recommend song", type="primary", use_container_width=True, disabled=(cur_mood is None)):
        if cur_mood:
            play_music_for_mood(cur_mood)
            st.rerun()


# 3. Right Column (Recommended Song, Player, Video)
with col_right:
    # Header
    st.subheader("Recommended Song") # Fixed casing title
    
    if st.session_state.current_song:
        song = st.session_state.current_song
        
        # Spacer to align with Camera Input box (which is below Radio buttons)
        st.markdown("<div style='height: 48px;'></div>", unsafe_allow_html=True)

        # Player Area
        with st.container(border=True):
            r1_col1, r1_col2 = st.columns([1, 2.5]) # Adjusted ratio for better right space
            
            with r1_col1:
                # Song Logo - Left Side
                st.image(song['cover_url'], use_container_width=True)
            
            with r1_col2:
                # Right Side: Title, Controls, Buttons
                st.markdown(f"<div class='song-title'>{song['title']}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='artist-name'>{song['artist']}</div>", unsafe_allow_html=True)
                
                # Progress Bar / Slider Adjusted
                pt_c1, pt_c2, pt_c3 = st.columns([0.8, 3, 0.8])
                
                progress = st.session_state.start_time
                with pt_c1:
                    st.write(f"{progress//60}:{progress%60:02d}")
                with pt_c2:
                    new_prog = st.slider("Progress", 0, 180, progress, label_visibility="collapsed")
                    if new_prog != progress:
                        st.session_state.start_time = new_prog
                        st.rerun()
                with pt_c3:
                     st.write("3:00")

                st.markdown("<br>", unsafe_allow_html=True) # Spacer

                # Control Buttons - NOW ON THE RIGHT SIDE
                b1, b2, b3, b4, b5, b6 = st.columns(6, gap="small")
                with b1:
                    if st.button("❤️", help="Like (+10)"): handle_feedback_and_next("Like")
                with b2:
                    if st.button("⏭️ 5s", help="Skip Song (Listened 5s)"): handle_feedback_and_next("skip_immediate")
                with b3:
                    if st.button("⏭️ Mid", help="Skip Song (Listened Half)"): handle_feedback_and_next("skip_mid")
                with b4:
                    if st.button("👎", help="Dislike (-15)"): handle_feedback_and_next("Dislike")
                with b5:
                     if st.button("➕", help="Add to Playlist (+20)"): 
                         if not any(item['title'] == song['title'] for item in st.session_state.playlist):
                            st.session_state.playlist.append(song)
                            # RL Logic: Adding to playlist is a HUGE compliment (+20 reward)
                            # We use a special handler that DOES NOT skip the song
                            handle_feedback_only("add_to_playlist")
                         else: st.toast("Already in Playlist")
                with b6:
                    current_m = st.session_state.detected_mood or "Current Mood"
                    if st.button("🤔", help=f"Not for {current_m.title()} (-10)"): handle_feedback_and_next("Wrong Vibe")

    # Show Video Option
    st.markdown("---")


    show_vid_btn = st.button(f"{'Hide' if st.session_state.show_video else 'Show'} video")
    if show_vid_btn:
        st.session_state.show_video = not st.session_state.show_video
        st.rerun()

    # Video Window - HIDDEN if False
    if st.session_state.show_video and st.session_state.current_song:
         with st.container(border=True, height=300):
             # Video Player
             video_url = song['link'].replace("music.youtube.com", "www.youtube.com")
             st.video(video_url, start_time=st.session_state.start_time, autoplay=True)
