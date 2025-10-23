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
    get_team_defense_rank_vs_position
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

# Load players
@st.cache_data(ttl=3600)
def load_players():
    players_df = get_all_active_players()
    active_players_df = players_df[players_df['is_active'] == True]
    return active_players_df

try:
    players_df = load_players()
    player_names = sorted(players_df['full_name'].tolist())
except Exception as e:
    st.error(f"Error loading players: {e}")
    player_names = []

# Player selection
st.sidebar.subheader("🎯 Select Player")
selected_player = st.sidebar.selectbox(
    "Choose a player",
    options=player_names,
    index=0 if player_names else None
)

# Stat selection
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

# Opponent selection
st.sidebar.subheader("🏟️ Matchup Info")

player_team = None
opponent_abbrev = None
player_position = 'F'

if selected_player:
    player_id = get_player_id(selected_player)
    if player_id:
        with st.spinner("Detecting player's team..."):
            player_team = get_player_current_team(player_id, season=current_season)
            if not player_team:
                player_team = get_player_current_team(player_id, season=prior_season)
            
            player_position = get_player_position(player_id, season=prior_season)
        
        if player_team:
            st.sidebar.info(f"**{selected_player}** plays for **{player_team}** ({player_position})")

# Get today's games
todays_games = get_todays_games()

# Filter games for player's team
relevant_games = []
if player_team and todays_games:
    for game in todays_games:
        if player_team in [game['home'], game['away']]:
            relevant_games.append(game)

# Show relevant games
if relevant_games:
    st.sidebar.success(f"🎯 **Next Game Found!**")
    for game in relevant_games:
        if player_team == game['home']:
            opponent_abbrev = game['away']
            st.sidebar.markdown(f"**{player_team}** vs **{opponent_abbrev}** (Home)")
        else:
            opponent_abbrev = game['home']
            st.sidebar.markdown(f"**{player_team}** @ **{opponent_abbrev}** (Away)")
    
    change_opponent = st.sidebar.checkbox("Choose different opponent")
    if change_opponent:
        opponent_abbrev = None
else:
    st.sidebar.info("📅 Select opponent below for next matchup")

# Manual opponent selection
if opponent_abbrev is None:
    st.sidebar.markdown("**Select Opponent:**")
    all_teams = get_all_nba_teams()
    
    if player_team:
        available_opponents = [team for team in all_teams if team != player_team]
    else:
        available_opponents = all_teams
    
    manual_opponent = st.sidebar.selectbox(
        "Choose opponent team",
        options=["-- Select Opponent --"] + available_opponents,
        help="Select the opposing team"
    )
    
    if manual_opponent != "-- Select Opponent --":
        opponent_abbrev = manual_opponent

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
    st.info("👈 Select a player from the sidebar to begin")

# Footer
st.markdown("---")
st.markdown("""
**Data Sources:** NBA.com (via nba_api) | **Model:** Enhanced Ridge Regression  
**Features:** Season blending, head-to-head history, opponent recent form, positional analysis  
**Note:** Projections are for informational purposes only. Always verify lines with your sportsbook.
""")