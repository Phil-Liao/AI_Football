"""UI-only prototype for the AI Football Premier League dashboard."""

import streamlit as st
from pathlib import Path

from storage import FixtureStorage


st.set_page_config(
    page_title="AI Football | Premier League",
    page_icon="⚽",
    layout="wide",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600;700;800&display=swap');
    :root { --ink: #102b23; --muted: #718078; --green: #176b4c; --line: #e5ebe5; }
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stApp { background: #f5f7f2; color: var(--ink); }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stSidebar"] { background: #102b23; }
    [data-testid="stSidebar"] * { color: #eff7ed !important; }
    .block-container { max-width: 1220px; padding-top: 1.5rem; padding-bottom: 3rem; }
    .topline { display:flex; justify-content:space-between; align-items:center; margin-bottom:2rem; }
    .brand { font-weight:800; font-size:1.05rem; letter-spacing:-.04em; color:#102b23; }
    .brand span { color:#247b53; }
    .league-pill { background:#e3f5df; color:#267443; border-radius:99px; padding:.42rem .75rem; font-size:.72rem; font-weight:700; letter-spacing:.08em; }
    .eyebrow { text-transform:uppercase; letter-spacing:.16em; font:500 .69rem 'DM Mono', monospace; color:#668071; }
    .hero-title { font-size:clamp(2.25rem, 5vw, 4.1rem); line-height:1.04; font-weight:800; letter-spacing:-.065em; color:#102b23; margin:.6rem 0 .8rem; }
    .hero-copy { color:#718078; font-size:1rem; margin-bottom:1.6rem; }
    .metric { background:white; border:1px solid #e5ebe5; border-radius:14px; padding:1.1rem 1.25rem; min-height:105px; }
    .metric-label { color:#718078; font-size:.75rem; text-transform:uppercase; letter-spacing:.11em; }
    .metric-value { color:#102b23; font-size:1.8rem; font-weight:800; letter-spacing:-.05em; margin-top:.25rem; }
    .section-head { display:flex; justify-content:space-between; align-items:end; margin:2.4rem 0 1rem; }
    .section-title { font-size:1.45rem; letter-spacing:-.045em; font-weight:800; margin:0; color:#102b23; }
    .match-card { display:flex; align-items:center; gap:1rem; background:#fff; border:1px solid #e5ebe5; border-radius:14px; padding:1rem 1.15rem; margin:.55rem 0; }
    .match-date { min-width:120px; color:#718078; font: .72rem 'DM Mono', monospace; }
    .teams { flex:1; font-weight:700; color:#19372b; }
    .score { min-width:65px; text-align:center; font-weight:800; font-size:1.05rem; color:#18382a; }
    .prediction { min-width:135px; text-align:right; color:#27734b; font-size:.79rem; font-weight:700; }
    .status { min-width:76px; text-align:center; color:#718078; font-size:.68rem; font-weight:700; text-transform:uppercase; letter-spacing:.1em; }
    .stButton > button[kind="primary"] { background:#176b4c; border-color:#176b4c; border-radius:9px; font-weight:700; }
    @media (max-width: 700px) {
      .match-card { gap:.45rem; padding:.8rem .6rem; flex-wrap:wrap; }
      .match-date { min-width:100%; }
      .teams { min-width:55%; }
      .prediction { min-width:100%; text-align:left; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## ⚽ Touchline AI")
    st.caption("PREMIER LEAGUE · GEMMA 4")
    st.divider()
    page = st.radio("Workspace", ["Matchday", "Leaderboard"], label_visibility="collapsed")
    st.divider()
    st.caption("PREDICTION MODEL")
    st.markdown("**Gemma 4**  \n`gemma-4-26b-a4b-it`")
    st.caption("A match prediction dashboard for the English Premier League.")

st.markdown(
    '<div class="topline"><div class="brand">TOUCHLINE <span>AI</span></div>'
    '<div class="league-pill">PREMIER LEAGUE</div></div>',
    unsafe_allow_html=True,
)
st.markdown('<div class="eyebrow">THE BEAUTIFUL GAME, WITH A SECOND OPINION</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-title">Every match has<br>another point of view.</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-copy">Gemma 4 calls the fixtures. The table keeps score.</div>', unsafe_allow_html=True)

toolbar = st.columns([1, 1, 2, 2])
with toolbar[0]:
    st.selectbox("Season", ["2025/26", "2024/25"])
with toolbar[1]:
    st.selectbox("Fixtures", ["All matches", "Upcoming", "Completed"])
with toolbar[3]:
    st.write("")
    st.button("↻  Refresh fixtures", type="primary", use_container_width=True)

metrics = st.columns(3)
for column, label, value in zip(
    metrics,
    ["Season fixtures", "Gemma predictions", "Bot points"],
    ["38", "—", "—"],
):
    with column:
        st.markdown(
            f'<div class="metric"><div class="metric-label">{label}</div>'
            f'<div class="metric-value">{value}</div></div>',
            unsafe_allow_html=True,
        )

if page == "Leaderboard":
    st.markdown('<div class="section-head"><p class="section-title">Bot leaderboard</p></div>', unsafe_allow_html=True)
    st.info("Leaderboard scores will appear here.")
else:
    heading, action = st.columns([3, 1])
    with heading:
        st.markdown('<div class="section-head"><p class="section-title">Matchday fixtures</p></div>', unsafe_allow_html=True)
    with action:
        st.write("")
        st.button("✦  Predict upcoming", use_container_width=True)

    fixtures = FixtureStorage(Path(__file__).with_name("data.csv")).load_matchday()
    if fixtures.empty:
        st.info("No stored fixtures found. Run football_data.py to fetch the matchday fixtures.")

    for fixture in fixtures.to_dict("records"):
        kickoff = fixture.get("match_date", "")
        home = fixture.get("home_team", "")
        away = fixture.get("away_team", "")
        home_score = fixture.get("home_score", "")
        away_score = fixture.get("away_score", "")
        score = (
            f"{home_score} – {away_score}"
            if str(home_score) not in ("", "nan") and str(away_score) not in ("", "nan")
            else "vs"
        )
        status = str(fixture.get("status", "")).upper() or "UPCOMING"
        prediction = "Prediction pending"
        st.markdown(
            '<div class="match-card">'
            f'<div class="match-date">{kickoff}</div>'
            f'<div class="teams">{home} <span style="color:#8d9a91;font-weight:400">vs</span> {away}</div>'
            f'<div class="score">{score}</div>'
            f'<div class="prediction">{prediction}</div>'
            f'<div class="status">{status}</div>'
            '</div>',
            unsafe_allow_html=True,
        )

st.markdown(
    '<div style="margin-top:2rem;color:#8a958d;font-size:.72rem">'
    'Sample dashboard · Premier League · Gemma 4</div>',
    unsafe_allow_html=True,
)
