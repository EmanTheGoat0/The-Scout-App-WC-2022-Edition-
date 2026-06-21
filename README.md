⚽ The Scout: AI Player Similarity Engine

The Scout is a highly interactive, data-driven web application that leverages professional football event data and Machine Learning to identify statistically similar players.

Built using open-source data from StatsBomb, this tool analyzes every single pass, shot, and tackle from the entire 64-match run of the 2022 FIFA World Cup, clustering player profiles based on their on-pitch actions and spatial behaviors.

✨ Features

🔍 AI Similarity Engine: Utilizes Scikit-Learn's Cosine Similarity and StandardScaler algorithms to find players performing similar tactical roles, mitigating raw volume bias.

📊 Comprehensive Feature Engineering: Aggregates granular, event-by-event data into robust statistical profiles (Pass Volume, Completion %, Shot Volume, Defensive Actions).

🗺️ Spatial Visualization (Pass Maps): Generates dynamic, coordinate-based pass maps using mplsoccer and matplotlib to visualize a player's exact passing footprint and progression tendencies.

⚡ Blazing Fast UI: Built entirely in Python with Streamlit, featuring @st.cache_data decorators to instantly query the heavy 64-match dataset after initial load.

🛠️ Tech Stack

Category

Technologies Used

Backend & ML

Python, Scikit-Learn, Pandas

Frontend UI

Streamlit

Data Source

StatsBomb API (statsbombpy)

Data Viz

Matplotlib, mplsoccer

🚀 How to Run Locally

If you want to clone this repository and run the similarity engine on your own machine, follow these steps:

1. Clone the repository:

git clone [https://github.com/EmanTheGoat0/the-scout-app.git](https://github.com/EmanTheGoat0/the-scout-app.git)
cd the-scout-app


2. Create and activate a virtual environment:

python -m venv venv
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate


3. Install the dependencies:

pip install -r requirements.txt


4. Launch the Streamlit app:

streamlit run app.py


📈 Future Roadmap / Next Steps

[ ] Expand the dataset to include the UEFA Champions League or FA Women's Super League.

[ ] Integrate K-Means Clustering to group players into new "Data-Driven Roles" (e.g., "Deep-Lying Playmaker" vs "Box-to-Box Destroyer").

[ ] Add defensive heatmaps (tackles and interceptions) alongside the pass maps.

Created by Emmanuel Falola.