from statsbombpy import sb
import pandas as pd

# Set pandas to show all columns
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

match_id_to_explore = 3857271 # England vs Iran

print(f"Fetching events for match ID {match_id_to_explore}...\n")

# This pulls every single action that happened in the match
events = sb.events(match_id=match_id_to_explore)

# Let's filter the data to ONLY look at passes
passes = events[events['type'] == 'Pass']

# Select the most useful columns to print
columns_to_show = ['minute', 'second', 'team', 'player', 'pass_recipient', 'location', 'pass_end_location']
filtered_passes = passes[columns_to_show]

print("Here are the first 15 passes of the match:\n")
print(filtered_passes.head(15))

print("\n---")
print("Notice the 'location' and 'pass_end_location' columns.")
print("Those are the [x, y] coordinates we need for heatmaps and advanced analytics!")