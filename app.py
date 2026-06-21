import streamlit as st
import pandas as pd
from statsbombpy import sb
import functools
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from mplsoccer import Pitch
import os
import ast

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="The Scout AI", layout="wide")
st.title("⚽ The Scout: AI Player Similarity Engine")
st.markdown("Analyzing aggregated data from **the entire 2022 FIFA World Cup**.")

# --- PHASE 1: LOAD DATA ---
@st.cache_resource(show_spinner=False)
def load_tournament_data():
    if os.path.exists("world_cup_data.csv"):
        return pd.read_csv("world_cup_data.csv", low_memory=False)
    
    matches = sb.matches(competition_id=43, season_id=106)
    match_ids = matches['match_id'].tolist()
    
    all_events = []
    progress_bar = st.progress(0, text="Downloading match data...")
    
    for i, mid in enumerate(match_ids):
        match_df = sb.events(match_id=mid)
        
        # CHANGE 2: ONLY keep essential columns to avoid crashing the cache
        cols = ['id', 'type', 'pass_outcome', 'player', 'team', 'location', 'pass_end_location']
        match_df = match_df[[c for c in cols if c in match_df.columns]]
        
        # Safely force coordinates to strings (leaving NaNs alone)
        if 'location' in match_df.columns:
            match_df['location'] = match_df['location'].apply(lambda x: str(x) if isinstance(x, list) else x)
        if 'pass_end_location' in match_df.columns:
            match_df['pass_end_location'] = match_df['pass_end_location'].apply(lambda x: str(x) if isinstance(x, list) else x)
            
        all_events.append(match_df)
        progress_bar.progress((i + 1) / len(match_ids), text=f"Match {i+1} of {len(match_ids)}...")
        
    progress_bar.empty()
    df = pd.concat(all_events, ignore_index=True)
    
    df.to_csv("world_cup_data.csv", index=False)
    return df

events = load_tournament_data()

# --- PHASE 2: FEATURE ENGINEERING ---
@st.cache_resource
def engineer_features(_events_df):
    passes = _events_df[_events_df['type'] == 'Pass'].copy()
    shots = _events_df[_events_df['type'] == 'Shot'].copy()
    tackles = _events_df[_events_df['type'] == 'Tackle'].copy()
    
    passes['is_successful'] = passes['pass_outcome'].isna()
    
    pass_stats = passes.groupby(['player', 'team']).agg(
        total_passes=('id', 'count'), successful_passes=('is_successful', 'sum')
    ).reset_index()
    shot_stats = shots.groupby(['player', 'team']).agg(total_shots=('id', 'count')).reset_index()
    tackle_stats = tackles.groupby(['player', 'team']).agg(total_tackles=('id', 'count')).reset_index()

    dfs = [pass_stats, shot_stats, tackle_stats]
    profiles = functools.reduce(lambda left, right: pd.merge(left, right, on=['player', 'team'], how='outer'), dfs).fillna(0)
    profiles['pass_completion_pct'] = (profiles['successful_passes'] / profiles['total_passes'] * 100).fillna(0).round(1)
    
    profiles = profiles[profiles['total_passes'] > 20]
    return profiles, passes

player_profiles, passes_df = engineer_features(events)

# --- UI & PHASE 3: MACHINE LEARNING ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🔍 Find Similar Players")
    player_list = sorted(player_profiles['player'].tolist())
    target_player = st.selectbox("Select a Player to Analyze:", player_list, index=player_list.index('Jude Bellingham') if 'Jude Bellingham' in player_list else 0)
    
    features = ['total_passes', 'pass_completion_pct', 'total_shots', 'total_tackles']
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(player_profiles[features])
    similarity_matrix = cosine_similarity(scaled_features)
    
    target_idx = player_profiles[player_profiles['player'] == target_player].index[0]
    player_profiles['similarity_score'] = similarity_matrix[target_idx]
    
    similar_players = player_profiles.sort_values(by='similarity_score', ascending=False).head(6)
    
    st.markdown(f"**Top 5 matches for {target_player}:**")
    display_df = similar_players[['player', 'team', 'similarity_score', 'total_passes', 'pass_completion_pct']].copy()
    display_df['similarity_score'] = (display_df['similarity_score'] * 100).round(1).astype(str) + "%"
    st.dataframe(display_df.iloc[1:], hide_index=True)

with col2:
    st.subheader(f"🗺️ {target_player}'s Tournament Pass Map")
    
    player_passes = passes_df[(passes_df['player'] == target_player) & (passes_df['is_successful'] == True)].copy()
    player_passes = player_passes.dropna(subset=['location', 'pass_end_location'])
    
    if len(player_passes) > 0:
        
        # Safe evaluation function to guarantee it never crashes on bad strings
        def safe_eval(val):
            try:
                return ast.literal_eval(val)
            except:
                return [0, 0]

        if isinstance(player_passes['location'].iloc[0], str):
            player_passes['location'] = player_passes['location'].apply(safe_eval)
            player_passes['pass_end_location'] = player_passes['pass_end_location'].apply(safe_eval)

        player_passes['x'] = player_passes['location'].apply(lambda loc: loc[0])
        player_passes['y'] = player_passes['location'].apply(lambda loc: loc[1])
        player_passes['end_x'] = player_passes['pass_end_location'].apply(lambda loc: loc[0])
        player_passes['end_y'] = player_passes['pass_end_location'].apply(lambda loc: loc[1])
        
        pitch = Pitch(pitch_type='statsbomb', pitch_color='#22312b', line_color='#c7d5cc')
        fig, ax = pitch.draw(figsize=(8, 6))
        
        pitch.lines(player_passes.x, player_passes.y,
                    player_passes.end_x, player_passes.end_y,
                    lw=2, transparent=True, comet=True, ax=ax, color='#ad993c')
        pitch.scatter(player_passes.x, player_passes.y, s=50, ax=ax, color='#ad993c', zorder=2)
        
        st.pyplot(fig)
    else:
        st.warning("Not enough pass data to generate a map.")