import streamlit as st
import pandas as pd
from statsbombpy import sb
import functools
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from mplsoccer import Pitch

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="The Scout AI", layout="wide")
st.title("⚽ The Scout: AI Player Similarity Engine")
st.markdown("Analyzing aggregated data from **the entire 2022 FIFA World Cup**.")

# --- PHASE 1: LOAD MULTIPLE MATCHES (Cached for speed) ---
@st.cache_data(show_spinner=False)
def load_tournament_data():
    matches = sb.matches(competition_id=43, season_id=106)
    
    # Load all 64 matches of the World Cup
    match_ids = matches['match_id'].tolist()
    
    all_events = []
    progress_bar = st.progress(0, text="Downloading match data from StatsBomb API...")
    
    for i, mid in enumerate(match_ids):
        all_events.append(sb.events(match_id=mid))
        progress_bar.progress((i + 1) / len(match_ids), text=f"Processing match {i+1} of {len(match_ids)}...")
        
    progress_bar.empty()
    return pd.concat(all_events, ignore_index=True)

events = load_tournament_data()

# --- PHASE 2: FEATURE ENGINEERING (Aggregated across tournament) ---
@st.cache_data
def engineer_features(events_df):
    passes = events_df[events_df['type'] == 'Pass'].copy()
    shots = events_df[events_df['type'] == 'Shot'].copy()
    tackles = events_df[events_df['type'] == 'Tackle'].copy()
    
    passes['is_successful'] = passes['pass_outcome'].isna()
    
    pass_stats = passes.groupby(['player', 'team']).agg(
        total_passes=('id', 'count'), successful_passes=('is_successful', 'sum')
    ).reset_index()
    shot_stats = shots.groupby(['player', 'team']).agg(total_shots=('id', 'count')).reset_index()
    tackle_stats = tackles.groupby(['player', 'team']).agg(total_tackles=('id', 'count')).reset_index()

    dfs = [pass_stats, shot_stats, tackle_stats]
    profiles = functools.reduce(lambda left, right: pd.merge(left, right, on=['player', 'team'], how='outer'), dfs).fillna(0)
    profiles['pass_completion_pct'] = (profiles['successful_passes'] / profiles['total_passes'] * 100).fillna(0).round(1)
    
    # Filter out players with very few actions to avoid skewed data
    profiles = profiles[profiles['total_passes'] > 20]
    return profiles, passes

player_profiles, passes_df = engineer_features(events)

# --- UI & PHASE 3: MACHINE LEARNING ---
# Create two columns for the layout
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🔍 Find Similar Players")
    # Dropdown to select a player
    player_list = sorted(player_profiles['player'].tolist())
    target_player = st.selectbox("Select a Player to Analyze:", player_list, index=player_list.index('Jude Bellingham') if 'Jude Bellingham' in player_list else 0)
    
    # ML Algorithm
    features = ['total_passes', 'pass_completion_pct', 'total_shots', 'total_tackles']
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(player_profiles[features])
    similarity_matrix = cosine_similarity(scaled_features)
    
    target_idx = player_profiles[player_profiles['player'] == target_player].index[0]
    player_profiles['similarity_score'] = similarity_matrix[target_idx]
    
    similar_players = player_profiles.sort_values(by='similarity_score', ascending=False).head(6)
    
    st.markdown(f"**Top 5 matches for {target_player} based on tournament stats:**")
    # Clean up dataframe for display
    display_df = similar_players[['player', 'team', 'similarity_score', 'total_passes', 'pass_completion_pct']].copy()
    display_df['similarity_score'] = (display_df['similarity_score'] * 100).round(1).astype(str) + "%"
    st.dataframe(display_df.iloc[1:], hide_index=True) # Hide the #1 result because it's the player themselves

# --- PHASE 4: VISUALIZATION (Pass Map) ---
with col2:
    st.subheader(f"🗺️ {target_player}'s Tournament Pass Map")
    
    # Filter passes for the selected player
    player_passes = passes_df[(passes_df['player'] == target_player) & (passes_df['is_successful'] == True)].copy()
    player_passes = player_passes.dropna(subset=['location', 'pass_end_location'])
    
    if len(player_passes) > 0:
        # Extract X and Y coordinates
        player_passes['x'] = player_passes['location'].apply(lambda loc: loc[0])
        player_passes['y'] = player_passes['location'].apply(lambda loc: loc[1])
        player_passes['end_x'] = player_passes['pass_end_location'].apply(lambda loc: loc[0])
        player_passes['end_y'] = player_passes['pass_end_location'].apply(lambda loc: loc[1])
        
        # Draw the Pitch using mplsoccer
        pitch = Pitch(pitch_type='statsbomb', pitch_color='#22312b', line_color='#c7d5cc')
        fig, ax = pitch.draw(figsize=(8, 6))
        
        # Plot the passes
        pitch.lines(player_passes.x, player_passes.y,
                    player_passes.end_x, player_passes.end_y,
                    lw=2, transparent=True, comet=True, ax=ax, color='#ad993c')
        
        # Plot starting node
        pitch.scatter(player_passes.x, player_passes.y, s=50, ax=ax, color='#ad993c', zorder=2)
        
        st.pyplot(fig)
    else:
        st.warning("Not enough pass data to generate a map.")