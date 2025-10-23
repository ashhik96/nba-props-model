import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import requests

from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import playergamelog, leaguedashteamstats, teamgamelog
from nba_api.live.nba.endpoints import scoreboard

# Rate limiting
def rate_limit():
    time.sleep(0.6)

@st.cache_data(ttl=86400)
def get_all_active_players():
    """Get all active NBA players"""
    all_players = players.get_players()
    return pd.DataFrame(all_players)

def get_player_id(player_name):
    """Get player ID from name"""
    player = players.find_players_by_full_name(player_name)
    if player:
        return player[0]['id']
    return None

@st.cache_data(ttl=3600)
def get_player_game_logs_cached(player_id, season='2024-25'):
    """Get player game logs for a season with caching"""
    try:
        rate_limit()
        gamelog = playergamelog.PlayerGameLog(player_id=player_id, season=season, season_type_all_star='Regular Season')
        df = gamelog.get_data_frames()[0]
        return df
    except Exception as e:
        print(f"Error fetching game logs for player {player_id} in {season}: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_team_stats_cached(season='2024-25'):
    """Get team stats for a season with caching"""
    try:
        rate_limit()
        team_stats = leaguedashteamstats.LeagueDashTeamStats(
            season=season,
            season_type_all_star='Regular Season',
            per_mode_detailed='PerGame'
        )
        df = team_stats.get_data_frames()[0]
        return df
    except Exception as e:
        print(f"Error fetching team stats for {season}: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_opponent_recent_games(opponent_abbrev, season='2024-25', last_n=10):
    """Get opponent's recent games"""
    try:
        all_teams = teams.get_teams()
        team_info = [t for t in all_teams if t['abbreviation'] == opponent_abbrev]
        
        if not team_info:
            return pd.DataFrame()
        
        team_id = team_info[0]['id']
        
        rate_limit()
        gamelog = teamgamelog.TeamGameLog(team_id=team_id, season=season, season_type_all_star='Regular Season')
        df = gamelog.get_data_frames()[0]
        
        return df.head(last_n)
    except Exception as e:
        print(f"Error fetching opponent recent games: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_head_to_head_history(player_id, opponent_abbrev, seasons=['2024-25', '2023-24']):
    """Get player's head-to-head history against a specific team"""
    h2h_games = []
    
    for season in seasons:
        try:
            rate_limit()
            gamelog = playergamelog.PlayerGameLog(player_id=player_id, season=season, season_type_all_star='Regular Season')
            df = gamelog.get_data_frames()[0]
            
            if not df.empty and 'MATCHUP' in df.columns:
                opponent_games = df[df['MATCHUP'].str.contains(opponent_abbrev, na=False)]
                if not opponent_games.empty:
                    h2h_games.append(opponent_games)
        except Exception as e:
            print(f"Error fetching h2h for {season}: {e}")
            continue
    
    if h2h_games:
        return pd.concat(h2h_games, ignore_index=True)
    return pd.DataFrame()

def fetch_fanduel_lines(api_key=None):
    """Placeholder for FanDuel lines - requires Odds API key"""
    return {}

@st.cache_data(ttl=1800)
def get_todays_games():
    """Get today's NBA games"""
    try:
        board = scoreboard.ScoreBoard()
        games_data = board.get_dict()
        
        games = []
        if 'scoreboard' in games_data and 'games' in games_data['scoreboard']:
            for game in games_data['scoreboard']['games']:
                games.append({
                    'home': game.get('homeTeam', {}).get('teamTricode', ''),
                    'away': game.get('awayTeam', {}).get('teamTricode', '')
                })
        return games
    except Exception as e:
        print(f"Error fetching today's games: {e}")
        return []

def get_all_nba_teams():
    """Get list of all NBA team abbreviations"""
    all_teams = teams.get_teams()
    return sorted([team['abbreviation'] for team in all_teams])

@st.cache_data(ttl=3600)
def get_player_current_team(player_id, season='2025-26'):
    """Get player's current team from recent game logs"""
    try:
        rate_limit()
        gamelog = playergamelog.PlayerGameLog(player_id=player_id, season=season, season_type_all_star='Regular Season')
        df = gamelog.get_data_frames()[0]
        
        if not df.empty and 'MATCHUP' in df.columns:
            matchup = df.iloc[0]['MATCHUP']
            if ' vs. ' in matchup:
                return matchup.split(' vs. ')[0]
            elif ' @ ' in matchup:
                return matchup.split(' @ ')[0]
        return None
    except Exception as e:
        print(f"Error fetching player team: {e}")
        return None

@st.cache_data(ttl=3600)
def get_player_position(player_id, season='2024-25'):
    """Get player's position from game logs"""
    try:
        rate_limit()
        gamelog = playergamelog.PlayerGameLog(player_id=player_id, season=season, season_type_all_star='Regular Season')
        df = gamelog.get_data_frames()[0]
        
        # NBA API doesn't directly provide position in game logs
        # We'll infer from player stats or use a default
        # For now, return a generic position based on player name lookup
        player_info = players.find_player_by_id(player_id)
        if player_info:
            # Default to F for forwards, G for guards, C for centers
            # This is a simplified approach
            return 'F'  # Default to Forward
        return 'F'
    except Exception as e:
        print(f"Error fetching player position: {e}")
        return 'F'

@st.cache_data(ttl=1800)
def get_team_next_game(team_abbrev):
    """
    Get a team's next upcoming game by checking scoreboard
    Returns dict with opponent info or None
    """
    try:
        time.sleep(0.5)
        
        board = scoreboard.ScoreBoard()
        games_data = board.get_dict()
        
        if 'scoreboard' in games_data and 'games' in games_data['scoreboard']:
            games = games_data['scoreboard']['games']
            
            for game in games:
                home_team = game.get('homeTeam', {}).get('teamTricode', '')
                away_team = game.get('awayTeam', {}).get('teamTricode', '')
                
                if team_abbrev == home_team:
                    return {
                        'opponent': away_team,
                        'is_home': True,
                        'matchup_string': f"{away_team} @ {home_team}",
                        'found': True
                    }
                elif team_abbrev == away_team:
                    return {
                        'opponent': home_team,
                        'is_home': False,
                        'matchup_string': f"{away_team} @ {home_team}",
                        'found': True
                    }
        
        return None
        
    except Exception as e:
        print(f"Error fetching next game: {e}")
        return None

@st.cache_data(ttl=86400)  # Cache for 24 hours
def scrape_defense_vs_position():
    """
    Scrape defensive rankings vs position from HashtagBasketball
    Returns DataFrame with columns: Position, Team, Rank, PTS
    """
    try:
        url = "https://hashtagbasketball.com/nba-defense-vs-position"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        time.sleep(1.0)
        
        print("Fetching defense vs position data...")
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code != 200:
            print(f"Failed to fetch defense vs position: {response.status_code}")
            return pd.DataFrame()
        
        tables = pd.read_html(response.content)
        
        if len(tables) < 4:
            print("Expected table not found")
            return pd.DataFrame()
        
        # Get table 3 (the main data table)
        raw_df = tables[3].copy()
        
        # The table has Position in column 0, "TEAM RANK" combined in column 1
        df_clean = pd.DataFrame()
        df_clean['Position'] = raw_df.iloc[:, 0]
        
        # Split "TEAM RANK" (e.g., "BOS 1" -> Team="BOS", Rank=1)
        team_rank = raw_df.iloc[:, 1].astype(str).str.split(expand=True)
        df_clean['Team'] = team_rank[0]
        df_clean['Rank'] = team_rank[1].astype(int)
        
        # Get PTS for reference
        pts_column = raw_df.iloc[:, 2].astype(str).str.split(expand=True)
        df_clean['PTS'] = pts_column[0].astype(float)
        
        print(f"Successfully scraped defense vs position for {len(df_clean)} team-position combinations")
        
        return df_clean
        
    except Exception as e:
        print(f"Error scraping defense vs position: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def get_players_by_team(team_abbrev, season='2024-25'):
    """
    Get all players who have played for a specific team in a season
    Returns DataFrame with player names and IDs
    """
    try:
        all_teams = teams.get_teams()
        team_info = [t for t in all_teams if t['abbreviation'] == team_abbrev]
        
        if not team_info:
            return pd.DataFrame()
        
        team_id = team_info[0]['id']
        
        rate_limit()
        from nba_api.stats.endpoints import commonteamroster
        roster = commonteamroster.CommonTeamRoster(team_id=team_id, season=season)
        df = roster.get_data_frames()[0]
        
        if not df.empty:
            # Return player names and IDs
            players_list = []
            for _, row in df.iterrows():
                players_list.append({
                    'player_id': row['PLAYER_ID'],
                    'full_name': row['PLAYER'],
                    'position': row.get('POSITION', 'F')
                })
            return pd.DataFrame(players_list)
        
        return pd.DataFrame()
    except Exception as e:
        print(f"Error fetching team roster: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def get_players_by_team(team_abbrev, season='2024-25'):
    """
    Get all players who have played for a specific team in a season
    Returns DataFrame with player names and IDs
    """
    try:
        all_teams = teams.get_teams()
        team_info = [t for t in all_teams if t['abbreviation'] == team_abbrev]
        
        if not team_info:
            return pd.DataFrame()
        
        team_id = team_info[0]['id']
        
        rate_limit()
        from nba_api.stats.endpoints import commonteamroster
        roster = commonteamroster.CommonTeamRoster(team_id=team_id, season=season)
        df = roster.get_data_frames()[0]
        
        if not df.empty:
            # Return player names and IDs
            players_list = []
            for _, row in df.iterrows():
                players_list.append({
                    'player_id': row['PLAYER_ID'],
                    'full_name': row['PLAYER'],
                    'position': row.get('POSITION', 'F')
                })
            return pd.DataFrame(players_list)
        
        return pd.DataFrame()
    except Exception as e:
        print(f"Error fetching team roster: {e}")
        return pd.DataFrame()


def get_team_defense_rank_vs_position(team_abbrev, player_position, def_vs_pos_df):
    """
    Get a team's defensive rank against a specific position
    
    Args:
        team_abbrev: Team abbreviation (e.g., 'ORL')
        player_position: Player position ('G', 'F', or 'C')
        def_vs_pos_df: DataFrame from scrape_defense_vs_position()
    
    Returns:
        dict with rank, total_teams, percentile, and rating
    """
    if def_vs_pos_df.empty:
        return {'rank': 15, 'total': 30, 'percentile': 50, 'rating': 'Average'}
    
    # Map player positions to website's position categories
    position_mapping = {
        'G': ['PG', 'SG'],  # Guards
        'F': ['SF', 'PF'],  # Forwards  
        'C': ['C']          # Centers
    }
    
    positions_to_check = position_mapping.get(player_position, ['SF'])
    
    # Get ranks for all relevant positions
    ranks = []
    for pos in positions_to_check:
        team_data = def_vs_pos_df[
            (def_vs_pos_df['Team'] == team_abbrev) & 
            (def_vs_pos_df['Position'] == pos)
        ]
        
        if not team_data.empty:
            ranks.append(team_data.iloc[0]['Rank'])
    
    if not ranks:
        return {'rank': 15, 'total': 30, 'percentile': 50, 'rating': 'Average'}
    
    # Average the ranks if multiple positions
    avg_rank = sum(ranks) / len(ranks)
    
    # Calculate percentile (lower rank = better defense)
    percentile = ((30 - avg_rank) / 30) * 100
    
    # Determine rating
    if avg_rank <= 10:
        rating = 'Elite'
    elif avg_rank <= 15:
        rating = 'Above Average'
    elif avg_rank <= 20:
        rating = 'Average'
    else:
        rating = 'Below Average'
    
    return {
        'rank': int(avg_rank),
        'total': 30,
        'percentile': round(percentile, 1),
        'rating': rating,
        'positions_checked': positions_to_check
    }