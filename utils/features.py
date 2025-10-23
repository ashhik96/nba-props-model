import pandas as pd
import numpy as np
from datetime import datetime

# Team abbreviation to full name mapping
TEAM_ABBREV_TO_NAME = {
    'ATL': 'Atlanta Hawks',
    'BOS': 'Boston Celtics',
    'BKN': 'Brooklyn Nets',
    'CHA': 'Charlotte Hornets',
    'CHI': 'Chicago Bulls',
    'CLE': 'Cleveland Cavaliers',
    'DAL': 'Dallas Mavericks',
    'DEN': 'Denver Nuggets',
    'DET': 'Detroit Pistons',
    'GSW': 'Golden State Warriors',
    'HOU': 'Houston Rockets',
    'IND': 'Indiana Pacers',
    'LAC': 'LA Clippers',
    'LAL': 'Los Angeles Lakers',
    'MEM': 'Memphis Grizzlies',
    'MIA': 'Miami Heat',
    'MIL': 'Milwaukee Bucks',
    'MIN': 'Minnesota Timberwolves',
    'NOP': 'New Orleans Pelicans',
    'NYK': 'New York Knicks',
    'OKC': 'Oklahoma City Thunder',
    'ORL': 'Orlando Magic',
    'PHI': 'Philadelphia 76ers',
    'PHX': 'Phoenix Suns',
    'POR': 'Portland Trail Blazers',
    'SAC': 'Sacramento Kings',
    'SAS': 'San Antonio Spurs',
    'TOR': 'Toronto Raptors',
    'UTA': 'Utah Jazz',
    'WAS': 'Washington Wizards'
}

def calculate_season_averages(game_logs_df):
    """Calculate season-to-date averages"""
    if game_logs_df.empty:
        return {}
    
    stats = {
        'PTS_avg': game_logs_df['PTS'].mean(),
        'AST_avg': game_logs_df['AST'].mean(),
        'REB_avg': game_logs_df['REB'].mean(),
        'FG3M_avg': game_logs_df['FG3M'].mean(),
        'PRA_avg': (game_logs_df['PTS'] + game_logs_df['REB'] + game_logs_df['AST']).mean(),
        'MIN_avg': game_logs_df['MIN'].mean() if 'MIN' in game_logs_df.columns else 0
    }
    return stats

def calculate_last_n_average(game_logs_df, n=5):
    """Calculate last N games average"""
    if game_logs_df.empty or len(game_logs_df) < n:
        return calculate_season_averages(game_logs_df)
    
    last_n = game_logs_df.head(n)
    stats = {
        f'PTS_last{n}': last_n['PTS'].mean(),
        f'AST_last{n}': last_n['AST'].mean(),
        f'REB_last{n}': last_n['REB'].mean(),
        f'FG3M_last{n}': last_n['FG3M'].mean(),
        f'PRA_last{n}': (last_n['PTS'] + last_n['REB'] + last_n['AST']).mean(),
        f'MIN_last{n}': last_n['MIN'].mean() if 'MIN' in last_n.columns else 0
    }
    return stats

def calculate_double_double_probability(game_logs_df):
    """Calculate probability of double-double"""
    if game_logs_df.empty:
        return 0.0
    
    def is_double_double(row):
        double_stats = sum([
            row['PTS'] >= 10,
            row['REB'] >= 10,
            row['AST'] >= 10
        ])
        return double_stats >= 2
    
    game_logs_df['is_dd'] = game_logs_df.apply(is_double_double, axis=1)
    return game_logs_df['is_dd'].mean()

def calculate_hit_rate(game_logs_df, stat_column, over_under_line, last_n=None):
    """Calculate hit rate: % of times player went OVER the line"""
    if game_logs_df.empty:
        return 0.0
    
    if last_n:
        games = game_logs_df.head(last_n)
    else:
        games = game_logs_df
    
    if len(games) == 0:
        return 0.0
    
    hits = (games[stat_column] > over_under_line).sum()
    return hits / len(games)

def get_opponent_defensive_stats(opponent_abbrev, team_stats_df, recent_games_df=None):
    """
    Get comprehensive opponent defensive stats
    Now includes recent form weighting
    """
    if team_stats_df.empty:
        return {
            'def_rating': 110.0,
            'pace': 100.0,
            'pts_allowed': 110.0,
            'recent_def_rating': 110.0,
            'def_trend': 0.0
        }
    
    # Convert abbreviation to full team name
    opponent_full_name = TEAM_ABBREV_TO_NAME.get(opponent_abbrev, opponent_abbrev)
    
    # Determine team column - check both abbreviation and name
    team_col = None
    search_value = opponent_abbrev
    
    if 'TEAM_ABBREVIATION' in team_stats_df.columns:
        team_col = 'TEAM_ABBREVIATION'
        search_value = opponent_abbrev
    elif 'TEAM_NAME' in team_stats_df.columns:
        team_col = 'TEAM_NAME'
        search_value = opponent_full_name
    else:
        print(f"ERROR: No team identifier column found")
        return {
            'def_rating': 110.0,
            'pace': 100.0,
            'pts_allowed': 110.0,
            'recent_def_rating': 110.0,
            'def_trend': 0.0
        }
    
    # Filter for the opponent team
    opp_stats = team_stats_df[team_stats_df[team_col] == search_value]
    
    if opp_stats.empty:
        print(f"WARNING: {search_value} not found in {team_col}")
        return {
            'def_rating': 110.0,
            'pace': 100.0,
            'pts_allowed': 110.0,
            'recent_def_rating': 110.0,
            'def_trend': 0.0
        }
    
    team_row = opp_stats.iloc[0]
    
    # Season averages
    pts_allowed = team_row.get('OPP_PTS', team_row.get('PTS', 110.0))
    def_rating = team_row.get('DEF_RATING', pts_allowed)
    pace = team_row.get('PACE', 100.0)
    
    # Calculate recent form if available
    recent_def_rating = def_rating
    def_trend = 0.0
    
    if recent_games_df is not None and not recent_games_df.empty:
        if 'PTS' in recent_games_df.columns:
            recent_pts = recent_games_df['PTS'].mean()
            recent_def_rating = recent_pts * 1.05
            def_trend = pts_allowed - recent_def_rating
    
    print(f"Found {opponent_abbrev} ({opponent_full_name}): Season Def={def_rating:.1f}, Recent={recent_def_rating:.1f}, Trend={def_trend:+.1f}")
    
    return {
        'def_rating': def_rating,
        'pace': pace,
        'pts_allowed': pts_allowed,
        'recent_def_rating': recent_def_rating,
        'def_trend': def_trend
    }

def analyze_head_to_head_performance(h2h_games_df, stat_type):
    """Analyze player's historical performance vs this opponent"""
    if h2h_games_df.empty:
        return {
            'h2h_avg': 0.0,
            'h2h_games': 0,
            'h2h_trend': 0.0
        }
    
    stat_map = {
        'PTS': 'PTS',
        'AST': 'AST',
        'REB': 'REB',
        'FG3M': 'FG3M'
    }
    
    stat_col = stat_map.get(stat_type, 'PTS')
    
    if stat_col not in h2h_games_df.columns:
        return {
            'h2h_avg': 0.0,
            'h2h_games': 0,
            'h2h_trend': 0.0
        }
    
    # Historical average vs this team
    h2h_avg = h2h_games_df[stat_col].mean()
    h2h_games = len(h2h_games_df)
    
    # Calculate trend
    if h2h_games >= 3:
        recent_3 = h2h_games_df.head(3)[stat_col].mean()
        older_games = h2h_games_df.iloc[3:][stat_col].mean() if h2h_games > 3 else h2h_avg
        h2h_trend = recent_3 - older_games
    else:
        h2h_trend = 0.0
    
    print(f"Head-to-head {stat_col}: {h2h_games} games, avg={h2h_avg:.1f}, trend={h2h_trend:+.1f}")
    
    return {
        'h2h_avg': h2h_avg,
        'h2h_games': h2h_games,
        'h2h_trend': h2h_trend
    }

def is_back_to_back(game_logs_df, game_index=0):
    """Check if this is a back-to-back game"""
    if game_logs_df.empty or len(game_logs_df) < 2:
        return False
    
    if game_index >= len(game_logs_df) - 1:
        return False
    
    try:
        current_date = pd.to_datetime(game_logs_df.iloc[game_index]['GAME_DATE'])
        prev_date = pd.to_datetime(game_logs_df.iloc[game_index + 1]['GAME_DATE'])
        return (current_date - prev_date).days == 1
    except:
        return False

def calculate_rest_days(game_logs_df, game_index=0):
    """Calculate days of rest since last game"""
    if game_logs_df.empty or len(game_logs_df) < 2:
        return 3
    
    if game_index >= len(game_logs_df) - 1:
        return 3
    
    try:
        current_date = pd.to_datetime(game_logs_df.iloc[game_index]['GAME_DATE'])
        prev_date = pd.to_datetime(game_logs_df.iloc[game_index + 1]['GAME_DATE'])
        return (current_date - prev_date).days
    except:
        return 3

def blend_season_stats(current_logs, prior_logs, min_games_threshold=10):
    """
    Blend current and prior season stats based on games played
    """
    current_games = len(current_logs) if not current_logs.empty else 0
    
    if current_games == 0:
        weight_current = 0.0
        weight_prior = 1.0
    elif current_games < min_games_threshold:
        weight_current = current_games / min_games_threshold
        weight_prior = 1.0 - weight_current
    else:
        weight_current = 0.85
        weight_prior = 0.15
    
    return weight_current, weight_prior

def build_enhanced_feature_vector(
    player_game_logs,
    opponent_abbrev,
    team_stats_df,
    prior_season_logs=None,
    opponent_recent_games=None,
    head_to_head_games=None,
    player_position='F'
):
    """
    Build enhanced feature vector with:
    - Opponent recent form
    - Head-to-head history
    - Season blending
    """
    features = {}
    
    # Season blending
    weight_current, weight_prior = blend_season_stats(player_game_logs, prior_season_logs)
    
    # Current season stats
    current_stats = calculate_season_averages(player_game_logs)
    
    # Prior season stats
    if prior_season_logs is not None and not prior_season_logs.empty:
        prior_stats = calculate_season_averages(prior_season_logs)
    else:
        prior_stats = current_stats
    
    # Blend stats
    for stat in ['PTS_avg', 'AST_avg', 'REB_avg', 'FG3M_avg', 'PRA_avg']:
        current_val = current_stats.get(stat, 0)
        prior_val = prior_stats.get(stat, 0)
        features[stat] = (weight_current * current_val) + (weight_prior * prior_val)
    
    # Last 5 and Last 10
    if not player_game_logs.empty:
        last5 = calculate_last_n_average(player_game_logs, n=5)
        last10 = calculate_last_n_average(player_game_logs, n=10)
        features.update(last5)
        features.update(last10)
    else:
        if prior_season_logs is not None and not prior_season_logs.empty:
            last5 = calculate_last_n_average(prior_season_logs, n=5)
            last10 = calculate_last_n_average(prior_season_logs, n=10)
            features.update(last5)
            features.update(last10)
    
    # Prior season reference
    features['prior_PTS'] = prior_stats.get('PTS_avg', 0)
    features['prior_AST'] = prior_stats.get('AST_avg', 0)
    features['prior_REB'] = prior_stats.get('REB_avg', 0)
    features['prior_FG3M'] = prior_stats.get('FG3M_avg', 0)
    
    # Enhanced opponent stats
    enhanced_opp_stats = get_opponent_defensive_stats(
        opponent_abbrev,
        team_stats_df,
        recent_games_df=opponent_recent_games
    )
    
    features['opp_def_rating'] = enhanced_opp_stats['def_rating']
    features['opp_recent_def_rating'] = enhanced_opp_stats['recent_def_rating']
    features['opp_def_trend'] = enhanced_opp_stats['def_trend']
    features['opp_pace'] = enhanced_opp_stats['pace']
    features['opp_pts_allowed'] = enhanced_opp_stats['pts_allowed']
    
    # Head-to-head analysis
    if head_to_head_games is not None and not head_to_head_games.empty:
        for stat in ['PTS', 'AST', 'REB', 'FG3M']:
            h2h_stats = analyze_head_to_head_performance(head_to_head_games, stat)
            features[f'h2h_{stat}_avg'] = h2h_stats['h2h_avg']
            features[f'h2h_{stat}_games'] = h2h_stats['h2h_games']
            features[f'h2h_{stat}_trend'] = h2h_stats['h2h_trend']
    else:
        for stat in ['PTS', 'AST', 'REB', 'FG3M']:
            features[f'h2h_{stat}_avg'] = 0.0
            features[f'h2h_{stat}_games'] = 0
            features[f'h2h_{stat}_trend'] = 0.0
    
    # Game context
    features['is_back_to_back'] = 1 if is_back_to_back(player_game_logs) else 0
    features['rest_days'] = calculate_rest_days(player_game_logs)
    
    # Double-double probability
    if not player_game_logs.empty:
        features['dd_probability'] = calculate_double_double_probability(player_game_logs)
    else:
        features['dd_probability'] = 0.0
    
    # Metadata
    features['weight_current'] = weight_current
    features['weight_prior'] = weight_prior
    features['current_games_played'] = len(player_game_logs) if not player_game_logs.empty else 0
    features['player_position'] = player_position
    
    return features