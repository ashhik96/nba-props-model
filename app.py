import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import time
from datetime import datetime

# make sure we can import from utils/
sys.path.append(os.path.join(os.path.dirname(__file__), "utils"))

from utils.data_fetcher import (
    get_player_id,
    get_opponent_recent_games,
    get_head_to_head_history,
    get_player_position,
    get_team_defense_rank_vs_position,
    get_players_by_team,
    get_upcoming_games,
    fetch_fanduel_lines,      # live Odds API -> FanDuel lines
    get_event_id_for_game,    # resolve game -> event id
    get_player_fanduel_line,  # pull line for one player/stat
)

from utils.cached_data_fetcher import (
    get_player_game_logs_cached_db,
    get_team_stats_cached_db,
    scrape_defense_vs_position_cached_db,
)

from utils.database import get_cache_stats, clear_old_seasons

from utils.features import (
    build_enhanced_feature_vector,
)

from utils.model import PlayerPropModel


# ─────────────────────────────
# Page config
# ─────────────────────────────
st.set_page_config(
    page_title="NBA Player Props Model",
    page_icon="🏀",
    layout="wide",
)


# ─────────────────────────────
# Season helpers
# ─────────────────────────────
def get_current_nba_season():
    """
    Guess current NBA season like '2025-26'.
    If today is Oct or later, it's YEAR-(YEAR+1 short).
    Otherwise it's (YEAR-1)-(YEAR short).
    """
    now = datetime.now()
    yr = now.year
    mo = now.month
    if mo >= 10:
        return f"{yr}-{str(yr+1)[2:]}"
    else:
        return f"{yr-1}-{str(yr)[2:]}"


def get_prior_nba_season():
    cur = get_current_nba_season()
    start_year = int(cur.split('-')[0])
    return f"{start_year-1}-{str(start_year)[2:]}"


current_season = get_current_nba_season()
prior_season = get_prior_nba_season()


# ─────────────────────────────
# Model
# ─────────────────────────────
@st.cache_resource
def load_model():
    return PlayerPropModel(alpha=1.0)

model = load_model()


# ─────────────────────────────
# Stat options for dropdown
# ─────────────────────────────
STAT_OPTIONS = {
    "Points": "PTS",
    "Assists": "AST",
    "Rebounds": "REB",
    "Three-Pointers Made": "FG3M",
    "Points + Rebounds + Assists (PRA)": "PRA",
    "Double-Double Probability": "DD",
}


# ─────────────────────────────
# Utility helpers for sportsbook + display
# ─────────────────────────────
def defense_emoji(rank_num: int) -> str:
    """
    Visual difficulty emoji:
    - rank <=10 : tough defense (red)
    - rank <=20 : middling (yellow/orange)
    - else      : soft / target (green)
    """
    if rank_num <= 10:
        return "🔴"
    elif rank_num <= 20:
        return "🟡"
    else:
        return "🟢"


def calc_hit_rate(game_logs: pd.DataFrame, stat_col: str, line_value: float, window: int = 10):
    """
    % of last `window` games the player went OVER line_value for stat_col.
    If not enough data or no line, return None.
    """
    if game_logs is None or game_logs.empty:
        return None
    if line_value is None:
        return None

    # last N games (most recent at head() because nba_api returns reverse-chronological)
    recent = game_logs.head(window).copy()

    if stat_col == "PRA":
        if not {"PTS", "REB", "AST"}.issubset(recent.columns):
            return None
        recent_vals = recent["PTS"] + recent["REB"] + recent["AST"]
    else:
        if stat_col not in recent.columns:
            return None
        recent_vals = recent[stat_col]

    if len(recent_vals) == 0:
        return None

    hits = (recent_vals > line_value).sum()
    rate = hits / len(recent_vals)
    return rate * 100.0


def calc_edge(prediction: float, line_value: float):
    """
    Compare model projection to sportsbook line.
    Returns (edge_str, rec_text, ou_short)
    edge_str ~ "+2.1 (+9.4%)"
    rec_text ~ "✅ OVER looks good" / "❌ UNDER looks good" / "⚪ No clear edge"
    ou_short ~ "OVER", "UNDER", or "—"
    """
    if line_value is None:
        return ("—", "No line", "—")

    if line_value == 0:
        diff = prediction
        pct = 0.0
    else:
        diff = prediction - line_value
        pct = (diff / line_value) * 100.0

    if abs(diff) < 1.5:
        rec_text = "⚪ No clear edge"
        ou_short = "—"
    elif diff > 1.5:
        rec_text = "✅ OVER looks good"
        ou_short = "OVER"
    else:
        rec_text = "❌ UNDER looks good"
        ou_short = "UNDER"

    edge_str = f"{diff:+.1f} ({pct:+.1f}%)"
    return (edge_str, rec_text, ou_short)


# ─────────────────────────────
# Detailed Player View Renderer
# (used inside each expander)
# ─────────────────────────────
def render_player_detail_body(pdata, cur_season, prev_season):
    """
    The deep dive panel for a single player.
    Called inside each expander, after we build pdata in the loop.
    """
    player_name = pdata["player_name"]
    team_abbrev = pdata["team_abbrev"]
    player_pos = pdata["player_pos"]
    opponent_abbrev = pdata["opponent_abbrev"]
    current_logs = pdata["current_logs"]
    prior_logs = pdata["prior_logs"]
    h2h_history = pdata["h2h_history"]
    opp_def_rank = pdata["opp_def_rank"]
    features = pdata["features"]
    prediction = pdata["prediction"]
    stat_code = pdata["stat_code"]
    stat_display = pdata["stat_display"]

    # sportsbook extras we calculated in build loop
    fd_line_val = pdata["fd_line_val"]           # may be None
    hit_pct_val = pdata["hit_pct_val"]           # may be None
    edge_str = pdata["edge_str"]                 # string or "—"
    rec_text = pdata["rec_text"]                 # recommendation string

    # games played info
    has_current = not current_logs.empty
    has_prior = not prior_logs.empty
    current_games = len(current_logs) if has_current else 0
    prior_games = len(prior_logs) if has_prior else 0
    h2h_games = 0 if h2h_history is None or h2h_history.empty else len(h2h_history)

    # ---- Header metrics
    st.subheader(f"📊 Projections for {player_name} ↩")

    colA, colB, colC = st.columns(3)
    with colA:
        st.metric(f"{cur_season} Games", current_games)
    with colB:
        st.metric(f"{prev_season} Games", prior_games)
    with colC:
        st.metric(f"vs {opponent_abbrev} History", h2h_games)

    if current_games < 5:
        st.info(
            f"Only {current_games} games in {cur_season}. "
            f"We're leaning more on {prev_season} + head-to-head."
        )

    # ---- Opponent defense vs position
    st.markdown("---")

    position_desc = {
        'G': 'Guards (PG/SG)',
        'F': 'Forwards (SF/PF)',
        'C': 'Centers (C)',
    }.get(player_pos, f'{player_pos} Position')

    st.subheader(f"🛡️ {opponent_abbrev} Defense vs {position_desc}")

    col1, col2, col3 = st.columns(3)

    rank_val = opp_def_rank.get("rank", 15)
    rating_text = opp_def_rank.get("rating", "Average")
    percentile = opp_def_rank.get("percentile", 50.0)

    rating_lower = str(rating_text).lower()
    if "elite" in rating_lower or "above" in rating_lower:
        diff_emoji = "🔴"
    elif "average" in rating_lower and "above" not in rating_lower:
        diff_emoji = "🟡"
    else:
        diff_emoji = "🟢"

    with col1:
        st.metric(
            "Defensive Rank vs Position",
            f"{diff_emoji} #{rank_val} of 30",
            help=f"How {opponent_abbrev} guards this archetype ({player_pos})"
        )
    with col2:
        st.metric(
            "Matchup Difficulty",
            rating_text,
            help="Elite / Above Avg = tough. Below Avg = soft / target spot."
        )
    with col3:
        st.metric(
            "Defense Percentile",
            f"{percentile:.0f}%",
            help="Higher percentile = stronger defense overall."
        )

    if "elite" in rating_lower or "above" in rating_lower:
        st.info(
            f"🔴 Tough matchup: {opponent_abbrev} defends {player_pos} well. "
            "Unders / caution."
        )
    elif "below" in rating_lower:
        st.success(
            f"🟢 Favorable matchup: {opponent_abbrev} struggles "
            f"vs {player_pos}. Overs become more viable."
        )

    # ---- Projection / performance / context
    st.markdown("---")
    colP, colR, colCxt = st.columns([2, 2, 1])

    # Projection panel
    with colP:
        st.subheader("🎯 Model Projection")
        if stat_code == "DD":
            st.metric("Double-Double Probability", f"{prediction:.1f}%")
        else:
            st.metric(
                f"Projected {stat_display}",
                f"{prediction:.1f}"
            )

        if "pts_allowed" in opp_def_rank:
            st.caption(
                f"🛡️ Opp vs {player_pos}: "
                f"{opp_def_rank['pts_allowed']:.1f} pts allowed"
            )
        else:
            st.caption("🛡️ Opponent defense data unavailable")

        if h2h_games > 0 and stat_code != "DD":
            h2h_avg = features.get(f"h2h_{stat_code}_avg", 0)
            st.caption(
                f"📊 vs {opponent_abbrev} Avg: {h2h_avg:.1f} "
                f"({h2h_games} games)"
            )

    # Recent performance
    with colR:
        st.subheader("📈 Recent Performance")
        season_avg = features.get(f"{stat_code}_avg", 0)
        last5 = features.get(f"{stat_code}_last5", season_avg)
        last10 = features.get(f"{stat_code}_last10", season_avg)

        if stat_code != "DD":
            st.write(f"**Season Average:** {season_avg:.1f}")
            st.write(f"**Last 5 Games:** {last5:.1f}")
            st.write(f"**Last 10 Games:** {last10:.1f}")

            wc = features.get("weight_current", 0)
            wp = features.get("weight_prior", 1)
            st.caption(
                f"Blend: {wc*100:.0f}% {cur_season}, "
                f"{wp*100:.0f}% {prev_season}"
            )
        else:
            st.write(f"Chance at DD: {prediction:.1f}% (model)")

    # Context
    with colCxt:
        st.subheader("🏀 Context")
        rest_days = features.get("rest_days", 3)
        is_b2b = features.get("is_back_to_back", 0)
        st.write(f"**Rest Days:** {rest_days}")
        st.write(f"**Back-to-Back:** {'Yes' if is_b2b else 'No'}")
        st.write(f"**Opponent:** {opponent_abbrev}")

    # ---- Sportsbook Line / Hit Rate section
    st.markdown("---")
    st.subheader("📊 Sportsbook Line & Hit Rate")

    colL, colH = st.columns(2)

    with colL:
        st.markdown("**Line / Edge**")
        if stat_code == "DD":
            st.write("Most books don't post DD props here, so no line.")
        else:
            if fd_line_val is None:
                st.write("Line: —")
                st.write("Edge vs Line: —")
                st.caption("No line available for this player/stat.")
            else:
                st.write(f"Line: **{fd_line_val}**")
                st.write(f"Edge vs Line: **{edge_str}**")
                st.caption(rec_text)

    with colH:
        st.markdown("**Hit Rate (Last 10 Games)**")
        if stat_code == "DD":
            st.write("Hit%: —")
            st.caption("N/A for DD market here.")
        else:
            if hit_pct_val is None:
                st.write("Hit%: —")
                st.caption("We only compute this if we have a line.")
            else:
                st.write(f"Hit%: **{hit_pct_val:.0f}%**")
                st.caption(
                    "Hit% = % of recent games over that line. "
                    "Historical only."
                )

    # ---- Head to head deep dive
    if h2h_games > 0 and stat_code != "DD":
        st.markdown("---")
        st.subheader(f"🔥 Head-to-Head vs {opponent_abbrev}")

        h2h_avg = features.get(f"h2h_{stat_code}_avg", 0)
        h2h_trend = features.get(f"h2h_{stat_code}_trend", 0)

        colH2H1, colH2H2 = st.columns(2)
        with colH2H1:
            st.markdown("**Average vs Opponent**")
            st.markdown(f"### Avg: {h2h_avg:.1f} ({h2h_games} games)")
            diff = h2h_avg - features.get(f"{stat_code}_avg", 0)
            clr = "green" if diff > 0 else "red"
            st.markdown(f":{clr}[{diff:+.1f} vs season avg]")

        with colH2H2:
            st.markdown("**Recent Trend**")
            if abs(h2h_trend) > 1:
                trending_up = (h2h_trend > 0)
                trend_text = "📈 Trending UP" if trending_up else "📉 Trending DOWN"
                st.markdown(f"### {trend_text}")
                st.markdown(
                    f":{('green' if trending_up else 'red')}[{h2h_trend:+.1f}]"
                )
            else:
                st.markdown("### ➡️ Consistent")

        if not h2h_history.empty:
            st.markdown("**Recent Games vs Opponent:**")
            base_cols = ["GAME_DATE","MATCHUP","PTS","REB","AST","FG3M"]
            show_cols = [c for c in base_cols if c in h2h_history.columns]
            if show_cols:
                h2h_recent = h2h_history.head(5)[show_cols].copy()
                if {"PTS","REB","AST"}.issubset(h2h_recent.columns):
                    h2h_recent["PRA"] = (
                        h2h_recent["PTS"] +
                        h2h_recent["REB"] +
                        h2h_recent["AST"]
                    )
                st.dataframe(h2h_recent, use_container_width=True)

    # ---- Recent game log (last 10)
    st.markdown("---")
    st.subheader("📋 Recent Game Log (Last 10 Games)")

    # stitch 10 most recent between current and prior
    if has_current and len(current_logs) >= 10:
        last10_logs = current_logs.head(10)
        label_season = cur_season
    elif has_current and len(current_logs) < 10:
        need = 10 - len(current_logs)
        last10_logs = pd.concat(
            [current_logs, prior_logs.head(need)], ignore_index=True
        ).head(10)
        label_season = f"{cur_season} + {prev_season}"
    elif has_prior:
        last10_logs = prior_logs.head(10)
        label_season = prev_season
    else:
        last10_logs = pd.DataFrame()
        label_season = "N/A"

    st.caption(f"Showing games from: {label_season}")

    if not last10_logs.empty:
        display_cols = [
            "GAME_DATE","MATCHUP","MIN","PTS","REB","AST",
            "FG3M","FGA","FG_PCT"
        ]
        cols_avail = [c for c in display_cols if c in last10_logs.columns]
        if cols_avail:
            preview_df = last10_logs[cols_avail].copy()
            if {"PTS","REB","AST"}.issubset(preview_df.columns):
                preview_df["PRA"] = (
                    preview_df["PTS"] +
                    preview_df["REB"] +
                    preview_df["AST"]
                )
            st.dataframe(preview_df, use_container_width=True)


# ─────────────────────────────
# Build matchup table + live expanders
# ─────────────────────────────
def build_matchup_view(
    selected_game: dict,
    stat_code: str,
    stat_display: str,
    cur_season: str,
    prev_season: str,
    model_obj: PlayerPropModel,
):
    """
    Stream the matchup board IF we have a selected game.
    If there's no game selected yet, show landing / instructions.
    """

    # If user hasn't picked a game yet
    if not selected_game:
        st.title("🏀 NBA Player Props Projection Model")
        st.markdown(
            "Advanced predictions using historical data, matchup analysis, "
            "and head-to-head history"
        )
        st.markdown("---")
        st.subheader("👋 How to use this tool")
        st.markdown(
            """
1. **Pick a matchup** on the left sidebar under *Select Upcoming Game*  
2. **Pick a stat** (*Points*, *Rebounds*, *3PM*, etc.)  
3. Watch the **Matchup Board** fill in player by player  
4. Scroll down and **expand any player** to see the full deep dive (model projection, line vs projection edge, rest days, head-to-head, etc.)

No game is selected yet — choose one in the sidebar to start.
            """
        )
        return

    home_team = selected_game["home"]
    away_team = selected_game["away"]

    # Title / description
    st.title("🏀 NBA Player Props Projection Model")
    st.markdown(
        "Advanced predictions using historical data, matchup analysis, "
        "and head-to-head history"
    )

    st.subheader("🏟 Matchup Board")
    st.caption(
        "Quick view of projections, sportsbook line, model edge, hit rate, and "
        "defensive matchup for everyone in this game."
    )

    # shared data we reuse for all players
    def_vs_pos_df = scrape_defense_vs_position_cached_db()
    team_stats = get_team_stats_cached_db(season=prev_season)

    # fetch rosters
    home_roster = get_players_by_team(home_team, season=cur_season)
    if home_roster.empty:
        home_roster = get_players_by_team(home_team, season=prev_season)
    if not home_roster.empty:
        home_roster["team_abbrev"] = home_team

    away_roster = get_players_by_team(away_team, season=cur_season)
    if away_roster.empty:
        away_roster = get_players_by_team(away_team, season=prev_season)
    if not away_roster.empty:
        away_roster["team_abbrev"] = away_team

    if home_roster.empty and away_roster.empty:
        st.error("Couldn't load rosters for this matchup.")
        return

    combined_roster = pd.concat([home_roster, away_roster], ignore_index=True)
    combined_roster = combined_roster.drop_duplicates(subset=["player_id"])

    total_players = len(combined_roster)

    # Pre-fetch FanDuel lines for this matchup (one call per matchup)
    # If the Odds API credits die or something fails, these will just be {}
    event_id = get_event_id_for_game(home_team, away_team)
    if event_id:
        odds_data = fetch_fanduel_lines(event_id)
    else:
        odds_data = {}

    # placeholders for streaming UI
    table_placeholder = st.empty()         # board table so far
    status_placeholder = st.empty()        # "Loaded X/Y"
    st.markdown("---")
    st.subheader("📂 Player Breakdowns (click any name below to expand)")
    st.caption(
        f"You can start opening players right away. "
        f"We'll keep adding more below as they finish loading. "
        f"({cur_season} vs {prev_season}, matchup context, line edge, trends, etc.)"
    )
    expanders_placeholder = st.empty()     # list of expanders so far

    table_rows = []
    player_payloads = []

    # stream each player
    for _, prow in combined_roster.iterrows():
        player_name = prow["full_name"]
        pid = prow["player_id"]
        team_abbrev = prow["team_abbrev"]
        opponent_abbrev = away_team if team_abbrev == home_team else home_team

        # position
        player_pos = get_player_position(pid, season=prev_season)

        # logs
        current_logs = get_player_game_logs_cached_db(
            pid, player_name, season=cur_season
        )
        prior_logs = get_player_game_logs_cached_db(
            pid, player_name, season=prev_season
        )

        # opponent recent form
        opponent_recent = get_opponent_recent_games(
            opponent_abbrev,
            season=prev_season,
            last_n=10
        )

        # head-to-head
        h2h_history = get_head_to_head_history(
            pid,
            opponent_abbrev,
            seasons=[prev_season, "2023-24"]
        )

        # defense rank vs position
        opp_def_rank_info = get_team_defense_rank_vs_position(
            opponent_abbrev,
            player_pos,
            def_vs_pos_df
        )

        # features for model
        feat = build_enhanced_feature_vector(
            current_logs,
            opponent_abbrev,
            team_stats,
            prior_season_logs=prior_logs,
            opponent_recent_games=opponent_recent,
            head_to_head_games=h2h_history,
            player_position=player_pos
        )

        # model projection (stat-based)
        if stat_code == "DD":
            pred_val = model_obj.predict_double_double(feat) * 100.0  # %
            proj_display = f"{pred_val:.1f}%"
        else:
            pred_val = model_obj.predict(feat, stat_code)
            proj_display = f"{pred_val:.1f}"

        # sportsbook line for THIS player/stat from FanDuel odds
        if stat_code == "DD":
            fd_line_val = None
        else:
            fd_info = get_player_fanduel_line(player_name, stat_code, odds_data)
            fd_line_val = fd_info["line"] if fd_info else None

        # compute edge + hit rate
        if stat_code == "DD":
            hit_pct_val = None
            edge_str = "—"
            rec_text = "Most books don't post DD lines"
            ou_short = "—"
        else:
            if fd_line_val is not None:
                hit_pct_val = calc_hit_rate(
                    current_logs if not current_logs.empty else prior_logs,
                    stat_code,
                    fd_line_val,
                    window=10
                )
                edge_str, rec_text, ou_short = calc_edge(pred_val, fd_line_val)
            else:
                hit_pct_val = None
                edge_str, rec_text, ou_short = ("—", "No line", "—")

        # Opp Def Rank vs Position w/ emoji color
        rank_num = opp_def_rank_info.get("rank", 15)
        rating_txt = opp_def_rank_info.get("rating", "Average")
        d_emoji = defense_emoji(rank_num)
        opp_def_display = f"{d_emoji} #{rank_num} ({rating_txt})"

        # add row for this player into table
        table_rows.append({
            "Player": player_name,
            "Team/Pos": f"{team_abbrev} · {player_pos}",
            "Proj": proj_display,
            "Line": "—" if fd_line_val is None else fd_line_val,
            "O/U": ou_short,
            "Hit%": "—" if hit_pct_val is None else f"{hit_pct_val:.0f}%",
            "Opp Def Rank vs Position": opp_def_display,
        })

        # prepare data for expander
        pdata = {
            "player_name": player_name,
            "team_abbrev": team_abbrev,
            "player_pos": player_pos,
            "opponent_abbrev": opponent_abbrev,
            "current_logs": current_logs,
            "prior_logs": prior_logs,
            "h2h_history": h2h_history,
            "opp_def_rank": opp_def_rank_info,
            "features": feat,
            "prediction": pred_val,
            "stat_code": stat_code,
            "stat_display": stat_display,

            # sportsbook stuff to show in detail view
            "fd_line_val": fd_line_val,
            "hit_pct_val": hit_pct_val,
            "edge_str": edge_str,
            "rec_text": rec_text,
        }
        player_payloads.append(pdata)

        # re-render table so far
        running_df = pd.DataFrame(table_rows)
        table_placeholder.dataframe(running_df, use_container_width=True)

        # re-render ALL expanders so far
        with expanders_placeholder.container():
            for info in player_payloads:
                with st.expander(
                    f"{info['player_name']} ({info['team_abbrev']} · {info['player_pos']})",
                    expanded=False
                ):
                    render_player_detail_body(info, cur_season, prev_season)

        # status line
        status_placeholder.write(
            f"Loaded {len(table_rows)}/{total_players} players..."
        )

        # tiny delay so UI visibly streams
        time.sleep(0.05)

    # final status
    status_placeholder.success("✅ Done.")


# ─────────────────────────────
# Sidebar Controls
# ─────────────────────────────
st.sidebar.header("⚙️ Settings")

# Cache stats block
with st.sidebar.expander("💾 Cache Stats"):
    cache_stats = get_cache_stats()
    st.write(f"**Players cached:** {cache_stats.get('total_players', 0)}")
    st.write(f"**Games cached:** {cache_stats.get('total_games', 0):,}")
    st.write(f"**DB Size:** {cache_stats.get('db_size_mb', 0):.1f} MB")

    if st.button("🗑️ Clear Old Seasons"):
        clear_old_seasons([current_season, prior_season])
        st.success("Old seasons cleared!")
        st.rerun()

# Upcoming games dropdown with a default "Select" option
st.sidebar.subheader("📅 Select Upcoming Game")

upcoming_games = get_upcoming_games(days=7)
selected_game = None
game_map = {}

if upcoming_games:
    sidebar_options = ["-- Select a Game --"]
    for g in upcoming_games:
        # ex: "Sat, Oct 25 - CHI @ ORL (7:30 PM)"
        date_disp = g.get("date_display", "")
        tm = f" ({g['time_display']})" if g.get("time_display") else ""
        label = f"{date_disp} - {g['away']} @ {g['home']}{tm}"
        sidebar_options.append(label)
        game_map[label] = g

    picked_label = st.sidebar.selectbox(
        f"Upcoming games (next 7 days) - {len(upcoming_games)} found",
        options=sidebar_options,
        index=0,  # default to instruction line
    )

    if picked_label != "-- Select a Game --":
        selected_game = game_map[picked_label]
        st.sidebar.info(
            f"Matchup: {selected_game['away']} @ {selected_game['home']}"
        )
else:
    st.sidebar.warning("⚠️ No upcoming games in next 7 days.")
    picked_label = None
    selected_game = None

# Which stat to predict on the board
st.sidebar.subheader("📊 Stat to Project")
stat_display_list = list(STAT_OPTIONS.keys())
stat_display_choice = st.sidebar.selectbox(
    "Choose stat to preview on the board",
    options=stat_display_list,
    index=0,
)
stat_code_choice = STAT_OPTIONS[stat_display_choice]

# ─────────────────────────────
# Main render call
# ─────────────────────────────
build_matchup_view(
    selected_game=selected_game,
    stat_code=stat_code_choice,
    stat_display=stat_display_choice,
    cur_season=current_season,
    prev_season=prior_season,
    model_obj=model,
)

# footer
st.markdown("---")
st.markdown(
    "**Data Sources:** NBA.com (via nba_api) | "
    "**Model:** Enhanced Ridge Regression  \n"
    "**Features:** Season blending, H2H history, opponent recent form, "
    "positional defense  \n"
    "**Sportsbook Lines:** FanDuel via The Odds API  \n"
    "**Note:** Projections are informational only. "
    "Always verify lines and context."
)
