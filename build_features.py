from statsbombpy import sb
import pandas as pd
import functools
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# Set pandas to show all columns so it looks clean in the terminal
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print("Fetching events for England vs Iran...\n")
match_id = 3857271
events = sb.events(match_id=match_id)

# 1. Isolate passes, shots, and tackles for our feature engine
passes = events[events['type'] == 'Pass'].copy()
shots = events[events['type'] == 'Shot'].copy()
tackles = events[events['type'] == 'Tackle'].copy()

# 2. Determine if a pass was successful
# In StatsBomb data, if the 'pass_outcome' is NaN (blank), the pass was SUCCESSFUL
passes['is_successful'] = passes['pass_outcome'].isna()

print("Aggregating data to build Player Profiles...\n")

# 3. Group by player to get their total stats for each action
pass_stats = passes.groupby(['player', 'team']).agg(
    total_passes=('id', 'count'),
    successful_passes=('is_successful', 'sum')
).reset_index()

shot_stats = shots.groupby(['player', 'team']).agg(total_shots=('id', 'count')).reset_index()
tackle_stats = tackles.groupby(['player', 'team']).agg(total_tackles=('id', 'count')).reset_index()

# Merge all stats into one master DataFrame
dfs = [pass_stats, shot_stats, tackle_stats]
player_profiles = functools.reduce(lambda left, right: pd.merge(left, right, on=['player', 'team'], how='outer'), dfs)

# If a player has passes but no shots, fill the NaN with 0
player_profiles = player_profiles.fillna(0)

# 4. Calculate Pass Completion Percentage
player_profiles['pass_completion_pct'] = (player_profiles['successful_passes'] / player_profiles['total_passes']) * 100
player_profiles['pass_completion_pct'] = player_profiles['pass_completion_pct'].fillna(0).round(1)

print("=== PHASE 3: MACHINE LEARNING (COSINE SIMILARITY) ===")

# These are the columns the ML model will look at
features = ['total_passes', 'pass_completion_pct', 'total_shots', 'total_tackles']

# SCALE THE DATA: This ensures Passes (100+) don't overpower Shots (1-5) in the math
scaler = StandardScaler()
scaled_features = scaler.fit_transform(player_profiles[features])

# Calculate the Cosine Similarity matrix (how close is everyone to everyone else)
similarity_matrix = cosine_similarity(scaled_features)

# Let's find Jude Bellingham's exact index in the table
target_player = 'Jude Bellingham'

# Verify the player exists to prevent errors
if target_player in player_profiles['player'].values:
    target_idx = player_profiles[player_profiles['player'] == target_player].index[0]

    # Extract his similarity scores and attach them to the DataFrame
    player_profiles['similarity_score'] = similarity_matrix[target_idx]

    # Sort to find the highest match!
    similar_players = player_profiles.sort_values(by='similarity_score', ascending=False)

    print(f"\nPLAYERS MOST SIMILAR TO {target_player.upper()} (STATISTICALLY):")
    # We use .head(6) because the #1 match will be Bellingham himself (1.00 score)
    print(similar_players[['player', 'team', 'similarity_score']].head(6).to_string(index=False))
else:
    print(f"Could not find {target_player} in the dataset. Please check the spelling.")