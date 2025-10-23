import pandas as pd
import numpy as np
from nba_api.stats.endpoints import playergamelog, leaguegamefinder, teamgamelog, leaguedashteamstats
from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import commonteamroster
import requests
from datetime import datetime, timedelta
import time
import streamlit as st

def get_all_active_players():
    """Get list of all active NBA players"""
    all_players = players.get_players()
    return pd.DataFrame(all_players)

def get_player_id(player_name):
    """Get player ID from name"""
    all_players = players.get_players()
    player = [p for p in all_players if p['full_name'].lower() == player_name.lower()]
    if player:
        return player[0]['id']
    return None

def get_player_current_team(player_id, season='2025-26'):
    """Get the team abbreviation for a player's current team"""
    try:
        time.sleep(1.0)
        gamelog = playergamelog.PlayerGameLog(
            player_id=player_id,
            season=season,
            season_type_all_star='Regular Season'
        )
        df = gamelog.get_data_frames()[0]
        
        if not df.empty:
            first_matchup = df.iloc[0]['MATCHUP']
            team_abbrev = first_matchup.split()[0]
            return team_abbrev
        
        # Fallback: try last season
        gamelog = playergamelog.PlayerGameLog(
            player_id=player_id,
            season='2024-25',
            season_type_all_star='Regular Season'
        )
        df = gamelog.get_data_frames()[0]
        
        if not df.empty:
            first_matchup = df.iloc[0]['MATCHUP']
            team_abbrev = first_matchup.split()[0]
            return team_abbrev
            
        return None
    except Exception as e:
        print(f"Error getting player's team: {e}")
        return None

@st.cache_data(ttl=86400)
def get_player_game_logs_cached(player_id, season='2025-26'):
    """Fetch player game logs with caching"""
    return get_player_game_logs(player_id, season)

def get_player_game_logs(player_id, season='2025-26'):
    """Fetch player game logs for a season"""
    try:
        time.sleep(1.0)
        gamelog = playergamelog.PlayerGameLog(
            player_id=player_id,
            season=season,
            season_type_all_star='Regular Season'
        )
        df = gamelog.get_data_frames()[0]
        return df
    except Exception as e:
        print(f"Error fetching game logs for {season}: {e}")
        try:
            time.sleep(3.0)
            gamelog = playergamelog.PlayerGameLog(
                player_id=player_id,
                season=season,
                season_type_all_star='Regular Season'
            )
            df = gamelog.get_data_frames()[0]
            return df
        except Exception as e2:
            print(f"Retry failed: {e2}")
            return pd.DataFrame()

@st.cache_data(ttl=86400)
def get_team_stats_cached(season='2025-26'):
    """Get team stats with caching"""
    return get_team_stats(season)

def get_team_stats(season='2025-26'):
    """Get all team stats using efficient LeagueDashTeamStats endpoint"""
    try:
        time.sleep(1.0)
        
        team_stats = leaguedashteamstats.LeagueDashTeamStats(
            season=season,
            season_type_all_star='Regular Season',
            per_mode_detailed='PerGame'
        )
        
        df = team_stats.get_data_frames()[0]
        
        if not df.empty:
            print(f"Successfully fetched stats for {len(df)} teams from {season}")
            return df
        
        return pd.DataFrame()
        
    except Exception as e:
        print(f"Error fetching team stats for {season}: {e}")
        
        # Fallback to previous season
        try:
            print(f"Trying fallback to previous season...")
            time.sleep(2.0)
            
            fallback_season = "2024-25" if season == "2025-26" else "2023-24"
            team_stats = leaguedashteamstats.LeagueDashTeamStats(
                season=fallback_season,
                season_type_all_star='Regular Season',
                per_mode_detailed='PerGame'
            )
            
            df = team_stats.get_data_frames()[0]
            
            if not df.empty:
                print(f"Fallback successful! Using {fallback_season} data")
                return df
            
            return pd.DataFrame()
            
        except Exception as e2:
            print(f"Fallback also failed: {e2}")
            return pd.DataFrame()

@st.cache_data(ttl=86400)
def get_opponent_recent_games(opponent_abbrev, season='2024-25', last_n=10):
    """Get opponent's last N games for recent form analysis"""
    try:
        time.sleep(1.0)
        
        all_teams = teams.get_teams()
        team_info = [t for t in all_teams if t['abbreviation'] == opponent_abbrev]
        
        if not team_info:
            return pd.DataFrame()
        
        team_id = team_info[0]['id']
        
        gamelog = teamgamelog.TeamGameLog(team_id=team_id, season=season)
        df = gamelog.get_data_frames()[0]
        
        if not df.empty:
            recent = df.head(last_n)
            print(f"Fetched {len(recent)} recent games for {opponent_abbrev}")
            return recent
        
        return pd.DataFrame()
        
    except Exception as e:
        print(f"Error fetching recent games for {opponent_abbrev}: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=86400)
def get_head_to_head_history(player_id, opponent_abbrev, seasons=['2024-25', '2023-24', '2022-23']):
    """Get player's historical performance against specific opponent"""
    try:
        all_games = []
        
        for season in seasons:
            time.sleep(1.0)
            gamelog = playergamelog.PlayerGameLog(
                player_id=player_id,
                season=season,
                season_type_all_star='Regular Season'
            )
            df = gamelog.get_data_frames()[0]
            
            if not df.empty:
                opponent_games = df[df['MATCHUP'].str.contains(opponent_abbrev, na=False)]
                if not opponent_games.empty:
                    all_games.append(opponent_games)
        
        if all_games:
            result = pd.concat(all_games, ignore_index=True)
            print(f"Found {len(result)} head-to-head games vs {opponent_abbrev}")
            return result
        
        return pd.DataFrame()
        
    except Exception as e:
        print(f"Error fetching head-to-head history: {e}")
        return pd.DataFrame()

def get_player_position(player_id, season='2025-26'):
    """Determine player's position from their stats"""
    try:
        time.sleep(1.0)
        
        gamelog = playergamelog.PlayerGameLog(
            player_id=player_id,
            season=season,
            season_type_all_star='Regular Season'
        )
        df = gamelog.get_data_frames()[0]
        
        if df.empty:
            gamelog = playergamelog.PlayerGameLog(
                player_id=player_id,
                season='2024-25',
                season_type_all_star='Regular Season'
            )
            df = gamelog.get_data_frames()[0]
        
        if not df.empty:
            avg_reb = df['REB'].mean()
            avg_ast = df['AST'].mean()
            
            # Simple position estimation
            if avg_reb > 8 and avg_ast < 4:
                return 'C'  # Center
            elif avg_reb > 6 or (avg_reb > 5 and avg_ast < 5):
                return 'F'  # Forward
            else:
                return 'G'  # Guard
        
        return 'F'
        
    except Exception as e:
        print(f"Error determining position: {e}")
        return 'F'

@st.cache_data(ttl=3600)
def get_todays_games():
    """Get today's NBA games"""
    try:
        from nba_api.live.nba.endpoints import scoreboard
        
        games = scoreboard.ScoreBoard()
        games_data = games.get_dict()
        
        matchups = []
        if 'scoreboard' in games_data and 'games' in games_data['scoreboard']:
            for game in games_data['scoreboard']['games']:
                home_team = game['homeTeam']['teamTricode']
                away_team = game['awayTeam']['teamTricode']
                matchups.append({
                    'matchup': f"{away_team} @ {home_team}",
                    'away': away_team,
                    'home': home_team
                })
        
        return matchups
    except Exception as e:
        print(f"Error fetching today's games: {e}")
        return []

def get_all_nba_teams():
    """Get list of all NBA team abbreviations"""
    all_teams = teams.get_teams()
    team_abbrevs = sorted([team['abbreviation'] for team in all_teams])
    return team_abbrevs

def fetch_fanduel_lines(api_key, sport='basketball_nba'):
    """Fetch FanDuel odds from The Odds API"""
    try:
        url = f"https://api.the-odds-api.com/v4/sports/{sport}/odds"
        params = {
            'apiKey': api_key,
            'regions': 'us',
            'markets': 'player_points,player_assists,player_rebounds,player_threes',
            'bookmakers': 'fanduel'
        }
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error fetching odds: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error fetching FanDuel lines: {e}")
        return None

def fallback_balldontlie(player_name, season=2025):
    """Fallback to BallDontLie API if nba_api fails"""
    try:
        url = f"https://www.balldontlie.io/api/v1/stats"
        params = {
            'seasons[]': season,
            'per_page': 100
        }
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return pd.DataFrame(data['data'])
        return pd.DataFrame()
    except Exception as e:
        print(f"BallDontLie fallback error: {e}")
        return pd.DataFrame()