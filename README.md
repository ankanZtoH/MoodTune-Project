# MoodTunes AI: Reinforcement Learning Music Recommender

## 🎵 Project Overview
MoodTunes AI is a context-aware music recommendation system that uses **Reinforcement Learning (Q-Learning)** to adapt to a user's emotional state. Unlike traditional recommenders that rely on static playlists, this agent **learns** from user feedback (Likes, Skips, Super Likes) to optimize music choices over time.

It features a hybrid **Mood Perception System** using Computer Vision (Facial Expression) and NLP (Sentiment Analysis + Keyword Scanning) to determine the "State" of the environment.

---

## 🧠 The Core Logic (Reinforcement Learning)

The project is built on the standard RL cycle: **State $\to$ Action $\to$ Reward $\to$ Next State**.

### 1. State Space (The Input)
The "Environment" consists of **10 Discrete Emotional States**:
*   Happy, Sad, Energetic, Calm, Angry
*   Romantic, Tired, Depressed, Anxious, Party

### 2. Action Space (The Output)
The Agent can choose from **14 Music Categories**:
*   *Examples:* Sad Bollywood, Gym Phonk, Lo-Fi Study, Heavy Metal, Upbeat Pop...

### 3. The Algorithm: Q-Learning
We use a **Q-Table** (Shape: $10 \times 14$) to store the "value" of playing a specific Genre for a specific Mood.

**The Bellman Update Rule:**
$$Q(s,a) \leftarrow Q(s,a) + \alpha [R + \gamma \max Q(s', a') - Q(s,a)]$$

**Hyperparameters (`src/config.py`):**
*   **$\alpha$ (Learning Rate) = 0.1**: The agent learns gradually, not forgetting old lessons too fast.
*   **$\gamma$ (Discount Factor) = 0.9**: The agent cares about long-term satisfaction.
*   **$\epsilon$ (Exploration) = 0.2**: 20% of the time, it tries a random song to discover new favorites.

### 4. Reward System (The Feedback Loop)
User feedback is translated into numerical rewards to train the brain (`src/environment.py`):

| Feedback Button | Reward | Meaning |
| :--- | :--- | :--- |
| **🌟 Super Like** | **+30** | "I love this! Play more like this." |
| **➕ Add to Playlist** | **+20** | "I want to keep this forever." |
| **❤️ Like** | **+10** | "Good match." |
| **Passively Finished** | **+5** | "It was okay." |
| **⏭️ Skip (5s)** | **-5** | "Not interested right now." |
| **🤔 Wrong Vibe** | **-10** | "Good song, but WRONG context!" |
| **👎 Dislike** | **-15** | "I hate this song." |

---

## 👁️ Mood Perception System (`src/mood_detection.py`)

The system uses a **Hybrid Approach** to detect the user's state:

### A. Text Analysis (NLP)
1.  **Keyword Scanning (Heuristic):** Scans for high-confidence triggers.
    *   *Input:* "I **killed** my **enemy** today." $\to$ Detects **Angry** (Aggregation).
    *   *Input:* "I won the **victory**." $\to$ Detects **Energetic**.
    *   *Input:* "**Lonely** and **lost**." $\to$ Detects **Depressed**.
2.  **Sentiment Analysis (TextBlob):** If no keywords are found, it analyzes the emotional polarity (Positive/Negative) of the sentence.

### B. Visual Analysis (Computer Vision)
*   **DeepFace (Library):** Uses a CNN to analyze facial expressions from the webcam.
*   **Fallback:** Uses a deterministic hashing algorithm if the camera is unavailable, ensuring the app never crashes during demos.

---

## 🌍 The Environment Physics (`src/environment.py`)

The environment follows the **Iso-Principle** of Music Therapy to simulate realistic mood transitions.

**Transition Probabilities:**
1.  **Positive Feedback (Like):**
    *   *Sad/Angry* $\to$ **Calm (60%)**, Neutral (30%), Happy (10%). (Gradual improvement).
    *   *Happy/Party* $\to$ **Sustain (80%)**. (Keeps the vibe going).
2.  **Negative Feedback (Dislike):**
    *   Mood often worsens (e.g., Happy $\to$ Angry) due to frustration.

---

## 📂 File Structure

*   **`app.py`**: The main Streamlit web application. Handles the UI, Player, and Feedback buttons.
*   **`src/agent.py`**: The Brain. Contains the `QLearningAgent` class, `learn()` function, and file saving/loading.
*   **`src/environment.py`**: The World. Defines Rewards, States, and Transition Logic.
*   **`src/mood_detection.py`**: The Eyes & Ears. Handles Text and Image processing.
*   **`src/simulation.py`**: The Training Dummy. A "Simulated User" that runs 200 epochs at start to pre-train the brain.
*   **`src/music_api.py`**: Connects to YouTube to fetch real song links.

---

## 🚀 How to Run
1.  **Install Dependencies:** `pip install -r requirements.txt` (streamlit, numpy, pandas, altair, textblob, deepface).
2.  **Run App:** `streamlit run app.py`
3.  **Interact:**
    *   Select a mood (Text/Camera/Manual).
    *   Listen to the song.
    *   **Give Feedback** to train the agent!
    *   Watch the "Brain Maturity" grow in the sidebar.
