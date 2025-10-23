import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from utils.data_fetcher import (
    get_all_active_players,
    get_player_id,
    get_player_game_logs_cached,
    get_team_stats_cached,
    fetch_fanduel_lines,
    get_todays_games,
    get_all_nba_teams,
    get_player_current_team,
    get_opponent_recent_games,
    get_head_to_head_history,
    get_player_position,
    scrape_defense_vs_position,
    get_team_defense_rank_vs_position,
    get_players_by_team
)
from utils.features import (
    calculate_season_averages,
    calculate_last_n_average,
    calculate_hit_rate,
    build_enhanced_feature_vector,
    calculate_double_double_probability
)
from utils.model import PlayerPropModel

# Page config
st.set_page_config(
    page_title="NBA Player Props Model",
    page_icon="🏀",
    layout="wide"
)

# Title
st.title("🏀 NBA Player Props Projection Model")
st.markdown("### Advanced predictions using historical data, matchup analysis, and head-to-head history")

# Sidebar for inputs
st.sidebar.header("⚙️ Settings")

# API Key input
if 'odds_api_key' not in st.session_state:
    st.session_state.odds_api_key = ""

api_key = st.sidebar.text_input(
    "Odds API Key (optional)",
    value=st.session_state.odds_api_key,
    type="password",
    help="Enter your Odds API key to fetch FanDuel lines"
)
st.session_state.odds_api_key = api_key

# Initialize model
@st.cache_resource
def load_model():
    model = PlayerPropModel(alpha=1.0)
    return model

model = load_model()

# Season setup
current_season = "2025-26"  # Current season (just started)
prior_season = "2024-25"    # Last full season

# Get today's games first
todays_games = get_todays_games()

# STEP 1: Game Selection
st.sidebar.subheader("🏀 Select Upcoming Game")

selected_game = None
selected_team = None
opponent_abbrev = None

if todays_games:
    game_options = ["-- Select a Game --"]
    game_map = {}
    
    for idx, game in enumerate(todays_games):
        game_str = f"{game['away']} @ {game['home']}"
        game_options.append(game_str)
        game_map[game_str] = game
    
    selected_game_str = st.sidebar.selectbox(
        "Choose from today's games",
        options=game_options,
        help="Select a game to analyze"
    )
    
    if selected_game_str != "-- Select a Game --":
        selected_game = game_map[selected_game_str]
        
        # STEP 2: Team Selection (from the selected game)
        st.sidebar.subheader("🏟️ Select Team to Analyze")
        
        team_options = [selected_game['home'], selected_game['away']]
        selected_team = st.sidebar.radio(
            "Which team's player to analyze?",
            options=team_options,
            format_func=lambda x: f"🏠 {x}" if x == selected_game['home'] else f"✈️ {x}"
        )
        
        # Set opponent based on selected team
        if selected_team == selected_game['home']:
            opponent_abbrev = selected_game['away']
        else:
            opponent_abbrev = selected_game['home']
        
        st.sidebar.info(f"**Matchup:** {selected_team} vs {opponent_abbrev}")
else:
    st.sidebar.warning("⚠️ No games scheduled for today")
    st.sidebar.info("You can manually select teams below")
    
    # Manual team selection if no games
    st.sidebar.subheader("🏟️ Manual Team Selection")
    all_teams = get_all_nba_teams()
    
    selected_team = st.sidebar.selectbox(
        "Select team to analyze",
        options=["-- Select Team --"] + all_teams
    )
    
    if selected_team and selected_team != "-- Select Team --":
        available_opponents = [team for team in all_teams if team != selected_team]
        opponent_abbrev = st.sidebar.selectbox(
            "Select opponent",
            options=["-- Select Opponent --"] + available_opponents
        )
        if opponent_abbrev == "-- Select Opponent --":
            opponent_abbrev = None

# STEP 3: Player Selection (from the selected team)
selected_player = None
player_team = None
player_position = 'F'

if selected_team and selected_team != "-- Select Team --":
    st.sidebar.subheader("🎯 Select Player")
    player_team = selected_team
    
    with st.spinner(f"Loading players from {selected_team}..."):
        # Get players from the selected team
        team_players_df = get_players_by_team(selected_team, season=current_season)
        
        # If current season roster is empty, try prior season
        if team_players_df.empty:
            team_players_df = get_players_by_team(selected_team, season=prior_season)
        
        if not team_players_df.empty:
            player_names = sorted(team_players_df['full_name'].tolist())
            
            # Select first player as default
            selected_player = st.sidebar.selectbox(
                f"Choose player from {selected_team}",
                options=player_names,
                index=0,
                help="Players from the selected team"
            )
            
            # Get player info
            if selected_player:
                player_id = get_player_id(selected_player)
                if player_id:
                    player_position = get_player_position(player_id, season=prior_season)
                    st.sidebar.success(f"✅ {selected_player} ({player_position})")
        else:
            st.sidebar.error(f"Could not load roster for {selected_team}")
            # Fallback to all players
            try:
                all_players_df = get_all_active_players()
                active_players_df = all_players_df[all_players_df['is_active'] == True]
                player_names = sorted(active_players_df['full_name'].tolist())
                selected_player = st.sidebar.selectbox(
                    "Choose any player",
                    options=player_names,
                    index=0
                )
            except Exception as e:
                st.sidebar.error(f"Error loading players: {e}")

# Stat selection
st.sidebar.subheader("📊 Stat to Predict")
stat_options = {
    "Points": "PTS",
    "Assists": "AST",
    "Rebounds": "REB",
    "Three-Pointers Made": "FG3M",
    "Points + Rebounds + Assists (PRA)": "PRA",
    "Double-Double Probability": "DD"
}

selected_stat_display = st.sidebar.selectbox(
    "Choose stat to predict",
    options=list(stat_options.keys())
)
selected_stat = stat_options[selected_stat_display]

# Main content
if selected_player:
    st.header(f"📊 Projections for {selected_player}")
    
    player_id = get_player_id(selected_player)
    
    if player_id:
        if opponent_abbrev is None or opponent_abbrev == "-- Select Opponent --":
            st.warning("⚠️ Please select an opponent team to see projections")
        else:
            with st.spinner("Fetching comprehensive player and matchup data..."):
                # Fetch player game logs
                current_logs = get_player_game_logs_cached(player_id, season=current_season)
                prior_logs = get_player_game_logs_cached(player_id, season=prior_season)
                
                # Fetch team stats
                team_stats = get_team_stats_cached(season=prior_season)
                
                       
                # Fetch opponent recent games
                opponent_recent = get_opponent_recent_games(opponent_abbrev, season=prior_season, last_n=10)
                
                # Fetch head-to-head history
                h2h_history = get_head_to_head_history(
                    player_id,
                    opponent_abbrev,
                    seasons=[prior_season, '2023-24']
                )
                # Fetch defense vs position rankings
                def_vs_pos_df = scrape_defense_vs_position()
            # Check data availability
            has_current_data = not current_logs.empty
            has_prior_data = not prior_logs.empty
            
            if not has_current_data and not has_prior_data:
                st.error(f"No game data found for {selected_player}")
            else:
                # Show data status
                current_games = len(current_logs) if has_current_data else 0
                prior_games = len(prior_logs) if has_prior_data else 0
                h2h_games = len(h2h_history) if not h2h_history.empty else 0
                
                col_info1, col_info2, col_info3 = st.columns(3)
                with col_info1:
                    st.metric("2025-26 Games", current_games)
                with col_info2:
                    st.metric("2024-25 Games", prior_games)
                with col_info3:
                    st.metric(f"vs {opponent_abbrev} History", h2h_games)
                
                if current_games < 5:
                    st.info(f"ℹ️ Only {current_games} games in {current_season}. Using {prior_season} data heavily + head-to-head history.")
                
                # Show opponent defense vs position
                opp_def_rank = get_team_defense_rank_vs_position(
                    opponent_abbrev,
                    player_position,
                    def_vs_pos_df
                )

                st.markdown("---")
                # Create a readable position description
                position_desc = {
                    'G': 'Guards (PG/SG)',
                    'F': 'Forwards (SF/PF)',
                    'C': 'Centers (C)'
                }.get(player_position, f'{player_position} Position')

                st.subheader(f"🛡️ {opponent_abbrev} Defense vs {position_desc}")

                col_def1, col_def2, col_def3 = st.columns(3)

                with col_def1:
                    rank = opp_def_rank['rank']
                    # Color code based on rank
                    if rank <= 10:
                        rank_color = "red"  # Tough defense
                        emoji = "🔴"
                    elif rank <= 20:
                        rank_color = "orange"  # Average
                        emoji = "🟡"
                    else:
                        rank_color = "green"  # Weak defense
                        emoji = "🟢"
                    
                    st.metric(
                        "Defensive Rank vs Position",
                        f"{emoji} #{rank} of 30",
                        help=f"How {opponent_abbrev} ranks defending {player_position} position"
                    )

                with col_def2:
                    rating = opp_def_rank['rating']
                    st.metric(
                        "Matchup Difficulty",
                        rating,
                        help="Elite = Top 10, Above Avg = 11-15, Avg = 16-20, Below Avg = 21+"
                    )

                with col_def3:
                    percentile = opp_def_rank['percentile']
                    st.metric(
                        "Defense Percentile",
                        f"{percentile:.0f}%",
                        help="Higher = Better defense"
                    )

                # Add explanation for color coding
                if rank <= 10:
                    st.info(f"🔴 **Tough Matchup:** {opponent_abbrev} is a top-10 defense vs {player_position}. Consider UNDER or avoid.")
                elif rank >= 21:
                    st.success(f"🟢 **Favorable Matchup:** {opponent_abbrev} ranks bottom-10 vs {player_position}. Consider OVER!")

                # Build enhanced features
                features = build_enhanced_feature_vector(
                    current_logs,
                    opponent_abbrev,
                    team_stats,
                    prior_season_logs=prior_logs,
                    opponent_recent_games=opponent_recent,
                    head_to_head_games=h2h_history,
                    player_position=player_position
                )
                
                # Make prediction
                st.markdown("---")
                col1, col2, col3 = st.columns([2, 2, 1])
                
                with col1:
                    st.subheader("🎯 Model Projection")
                    
                    if selected_stat == "DD":
                        prediction = model.predict_double_double(features) * 100
                        st.metric(
                            label="Double-Double Probability",
                            value=f"{prediction:.1f}%"
                        )
                    else:
                        prediction = model.predict(features, selected_stat)
                        st.metric(
                            label=f"Projected {selected_stat_display}",
                            value=f"{prediction:.1f}"
                        )
                        
                        # Show key factors
                        st.caption(f"🛡️ Opp Recent Def: {features.get('opp_recent_def_rating', 110):.1f} ppg")
                        if h2h_games > 0:
                            h2h_avg = features.get(f'h2h_{selected_stat}_avg', 0)
                            st.caption(f"📊 vs {opponent_abbrev} Avg: {h2h_avg:.1f} ({h2h_games} games)")
                
                with col2:
                    st.subheader("📈 Recent Performance")
                    
                    season_avg = features.get(f'{selected_stat}_avg', 0)
                    last5_avg = features.get(f'{selected_stat}_last5', 0)
                    last10_avg = features.get(f'{selected_stat}_last10', 0)
                    
                    if selected_stat != "DD":
                        st.write(f"**Season Average:** {season_avg:.1f}")
                        st.write(f"**Last 5 Games:** {last5_avg:.1f}")
                        st.write(f"**Last 10 Games:** {last10_avg:.1f}")
                        
                        weight_current = features.get('weight_current', 0)
                        weight_prior = features.get('weight_prior', 1)
                        st.caption(f"Blend: {weight_current*100:.0f}% current, {weight_prior*100:.0f}% last season")
                
                with col3:
                    st.subheader("🏀 Context")
                    rest_days = features.get('rest_days', 3)
                    is_b2b = features.get('is_back_to_back', 0)
                    
                    st.write(f"**Rest Days:** {rest_days}")
                    st.write(f"**Back-to-Back:** {'Yes' if is_b2b else 'No'}")
                    st.write(f"**Opponent:** {opponent_abbrev}")
                
                # FanDuel Lines
                st.markdown("---")
                st.subheader("📊 FanDuel Line & Hit Rate Analysis")
                
                col_line1, col_line2 = st.columns(2)
                
                with col_line1:
                    manual_line = st.number_input(
                        f"Enter FanDuel O/U Line for {selected_stat_display}",
                        min_value=0.0,
                        max_value=200.0,
                        value=season_avg if selected_stat != "DD" else 0.0,
                        step=0.5
                    )
                
                with col_line2:
                    if selected_stat != "DD":
                        edge = prediction - manual_line
                        edge_pct = (edge / manual_line * 100) if manual_line > 0 else 0
                        
                        edge_color = "green" if edge > 0 else "red"
                        st.markdown(f"**Edge vs Line:** :{edge_color}[{edge:+.1f} ({edge_pct:+.1f}%)]")
                        
                        if abs(edge) < 1.5:
                            recommendation = "⚪ No clear edge"
                        elif edge > 1.5:
                            recommendation = "✅ OVER looks good"
                        else:
                            recommendation = "❌ UNDER looks good"
                        
                        st.markdown(f"**Recommendation:** {recommendation}")
                
                # Hit Rate Analysis
                display_logs = current_logs if has_current_data else prior_logs
                display_season = current_season if has_current_data else prior_season
                
                if selected_stat != "DD" and manual_line > 0 and not display_logs.empty:
                    st.markdown("---")
                    st.subheader(f"🎯 Hit Rate vs Current Line ({display_season})")
                    
                    col_hit1, col_hit2, col_hit3 = st.columns(3)
                    
                    stat_column = selected_stat
                    if selected_stat == "PRA":
                        display_logs['PRA'] = display_logs['PTS'] + display_logs['REB'] + display_logs['AST']
                        stat_column = 'PRA'
                    
                    season_hit_rate = calculate_hit_rate(display_logs, stat_column, manual_line)
                    last5_hit_rate = calculate_hit_rate(display_logs, stat_column, manual_line, last_n=5)
                    last10_hit_rate = calculate_hit_rate(display_logs, stat_column, manual_line, last_n=10)
                    
                    with col_hit1:
                        st.metric(
                            "Season Hit Rate (OVER)",
                            f"{season_hit_rate*100:.1f}%"
                        )
                    
                    with col_hit2:
                        st.metric(
                            "Last 5 Games Hit Rate",
                            f"{last5_hit_rate*100:.1f}%"
                        )
                    
                    with col_hit3:
                        st.metric(
                            "Last 10 Games Hit Rate",
                            f"{last10_hit_rate*100:.1f}%"
                        )
                
                # Head-to-Head Analysis
                if h2h_games > 0 and selected_stat != "DD":
                    st.markdown("---")
                    st.subheader(f"🔥 Head-to-Head vs {opponent_abbrev}")
                    
                    h2h_avg = features.get(f'h2h_{selected_stat}_avg', 0)
                    h2h_trend = features.get(f'h2h_{selected_stat}_trend', 0)
                    
                    col_h2h1, col_h2h2 = st.columns(2)
                    
                    with col_h2h1:
                        st.metric(
                            f"Career Avg vs {opponent_abbrev}",
                            f"{h2h_avg:.1f}",
                            delta=f"{h2h_avg - season_avg:+.1f} vs season avg"
                        )
                    
                    with col_h2h2:
                        if abs(h2h_trend) > 1:
                            trend_text = "Trending UP" if h2h_trend > 0 else "Trending DOWN"
                            st.metric(
                                "Recent Trend vs Them",
                                trend_text,
                                delta=f"{h2h_trend:+.1f}"
                            )
                        else:
                            st.metric("Recent Trend vs Them", "Consistent")
                    
                    # Show recent h2h games
                    if not h2h_history.empty:
                        st.write("**Recent Games vs Opponent:**")
                        h2h_display_cols = ['GAME_DATE', 'MATCHUP', 'PTS', 'REB', 'AST', 'FG3M']
                        available_h2h_cols = [col for col in h2h_display_cols if col in h2h_history.columns]
                        
                        if available_h2h_cols:
                            recent_h2h = h2h_history.head(5)[available_h2h_cols].copy()
                            if 'PTS' in recent_h2h.columns and 'REB' in recent_h2h.columns and 'AST' in recent_h2h.columns:
                                recent_h2h['PRA'] = recent_h2h['PTS'] + recent_h2h['REB'] + recent_h2h['AST']
                            st.dataframe(recent_h2h, use_container_width=True)
                
                # Game log table
                st.markdown("---")
                st.subheader(f"📋 Recent Game Log ({display_season})")
                
                display_cols = ['GAME_DATE', 'MATCHUP', 'MIN', 'PTS', 'REB', 'AST', 'FG3M', 'FGA', 'FG_PCT']
                available_cols = [col for col in display_cols if col in display_logs.columns]
                
                if available_cols:
                    recent_games = display_logs.head(10)[available_cols].copy()
                    
                    if 'PTS' in recent_games.columns and 'REB' in recent_games.columns and 'AST' in recent_games.columns:
                        recent_games['PRA'] = recent_games['PTS'] + recent_games['REB'] + recent_games['AST']
                    
                    st.dataframe(recent_games, use_container_width=True)
                
    else:
        st.error(f"Could not find player ID for {selected_player}")
else:
    st.info("👈 Select a game and team from the sidebar to begin")
    
    # Show helpful instructions
    st.markdown("""
    ### How to Use:
    1. **Select a Game** - Choose from today's scheduled games (if available)
    2. **Select a Team** - Pick which team to analyze from the matchup
    3. **Select a Player** - Choose a player from the selected team (default player auto-selected)
    4. **Choose a Stat** - Select the stat you want to predict
    5. **View Projections** - Get AI-powered predictions and analysis
    """)

# Footer
st.markdown("---")
st.markdown("""
**Data Sources:** NBA.com (via nba_api) | **Model:** Enhanced Ridge Regression  
**Features:** Season blending, head-to-head history, opponent recent form, positional analysis  
**Note:** Projections are for informational purposes only. Always verify lines with your sportsbook.
""")