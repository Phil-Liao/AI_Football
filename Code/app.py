"""UI-only prototype for the AI Football Premier League dashboard."""

from pathlib import Path

import pandas as pd
import requests
import streamlit as st

from main import MainApplication


st.set_page_config(
    page_title="AI Football | Premier League",
    page_icon="⚽",
    layout="wide",
)


@st.cache_resource
def _application() -> MainApplication:
    return MainApplication(Path(__file__).parent)


application = _application()
application.start_daily_predictions()

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
    .stButton > button { color:#102b23 !important; background:#e3f5df; border:1px solid #b8d8be; border-radius:9px; font-weight:700; }
    .stButton > button:hover { color:#102b23 !important; background:#d2ebcf; border-color:#8fbe9a; }
    .stButton > button:focus-visible { outline:3px solid #247b53; outline-offset:2px; }
    .stButton > button[kind="primary"] { color:#fff !important; background:#176b4c; border-color:#176b4c; border-radius:9px; font-weight:700; }
    .stButton > button[kind="primary"]:hover { color:#fff !important; background:#12583e; border-color:#12583e; }
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

st.markdown(
    '<div class="topline"><div class="brand">TOUCHLINE <span>AI</span></div>'
    '<div class="league-pill">PREMIER LEAGUE</div></div>',
    unsafe_allow_html=True,
)
st.markdown('<div class="eyebrow">THE BEAUTIFUL GAME, WITH A SECOND OPINION</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-title">Every match has<br>another point of view.</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-copy">Prediction models call the fixtures. The table keeps score.</div>', unsafe_allow_html=True)

toolbar = st.columns([1, 1, 2, 2])
with toolbar[0]:
    st.selectbox("Season", ["2025/26", "2024/25"])
with toolbar[1]:
    fixture_filter = st.selectbox("Fixtures", ["All matches", "Upcoming", "Completed"])
with toolbar[3]:
    st.write("")
    refresh_clicked = st.button(
        "↻  Refresh fixtures", type="primary", use_container_width=True
    )

if refresh_clicked:
    try:
        with st.spinner("Refreshing Premier League fixtures..."):
            _refreshed_matches, refreshed_count = application.refresh_fixtures()
    except (requests.exceptions.RequestException, ValueError) as error:
        st.error(f"Could not refresh fixtures: {error}")
    else:
        st.success(f"Refreshed {refreshed_count} Premier League fixtures.")

dashboard = application.load_dashboard()
all_fixtures = dashboard["fixtures"]
stored_predictions = dashboard["predictions"]
active_model_names = dashboard["active_model_names"]
model_summaries = dashboard.get("model_summaries", [])

with st.sidebar:
    st.markdown("## ⚽ Touchline AI")
    st.caption("PREMIER LEAGUE")
    st.divider()
    page = st.radio("Workspace", ["Matchday", "Leaderboard"], label_visibility="collapsed")
    st.divider()
    st.caption("PREDICTION MODELS")
    if model_summaries:
        for model in model_summaries:
            st.markdown(
                f"**{model['name']}**  \n`{model['model_id']}`"
            )
    else:
        st.caption("No prediction providers registered.")

prediction_count = sum(
    int(model["prediction_count"])
    for model in model_summaries
)
total_points = sum(
    int(model["points"])
    for model in model_summaries
)

metrics = st.columns(3)
for column, label, value in zip(
    metrics,
    ["Season fixtures", "Model predictions", "Bot points"],
    [str(len(all_fixtures)), str(prediction_count), str(total_points)],
):
    with column:
        st.markdown(
            f'<div class="metric"><div class="metric-label">{label}</div>'
            f'<div class="metric-value">{value}</div></div>',
            unsafe_allow_html=True,
        )

if page == "Leaderboard":
    st.markdown('<div class="section-head"><p class="section-title">Bot leaderboard</p></div>', unsafe_allow_html=True)
    leaderboard = dashboard["leaderboard"]
    if leaderboard.empty:
        st.info("Leaderboard scores will appear here after completed matches are scored.")
    else:
        display_rows = []
        for row in leaderboard.to_dict("records"):
            model_name = active_model_names.get(str(row["model"]), str(row["model"]))
            display_rows.append(
                {
                    "Model": model_name,
                    "Predictions": int(row["predictions"]),
                    "Correct": int(row["correct"]),
                    "Points": int(row["points"]),
                    "Accuracy": f'{float(row["accuracy"]):.1%}',
                }
            )
        st.dataframe(display_rows, use_container_width=True, hide_index=True)
else:
    all_fixtures = dashboard["fixtures"]
    fixtures = all_fixtures
    completed_statuses = {"complete", "completed", "finished"}
    statuses = fixtures["status"].fillna("").astype(str).str.strip().str.lower()
    if fixture_filter == "Upcoming":
        kickoff_times = pd.to_datetime(fixtures["match_date"], errors="coerce", utc=True)
        future_kickoffs = kickoff_times > pd.Timestamp.now(tz="UTC")
        fixtures = fixtures.loc[future_kickoffs & ~statuses.isin(completed_statuses)]
    elif fixture_filter == "Completed":
        fixtures = fixtures.loc[statuses.isin(completed_statuses)]

    stored_predictions = dashboard["predictions"]
    active_model_names = dashboard["active_model_names"]
    disk_predictions: dict[str, dict[str, str]] = {}
    for row in stored_predictions.to_dict("records"):
        match_id = str(row.get("match_id", ""))
        model_id = str(row.get("model", ""))
        prediction = str(row.get("prediction", "")).strip()
        if match_id and model_id:
            if prediction and not prediction.lower().startswith("prediction unavailable"):
                model_name = active_model_names.get(model_id, model_id)
                disk_predictions.setdefault(model_name, {})[match_id] = prediction

    heading, action = st.columns([3, 1])
    with heading:
        st.markdown('<div class="section-head"><p class="section-title">Matchday fixtures</p></div>', unsafe_allow_html=True)
    with action:
        st.write("")
        predict_clicked = st.button("✦  Predict missing", use_container_width=True, key="predict_upcoming")

    if predict_clicked:
        if all_fixtures.empty:
            st.warning("No stored fixtures found. Run football_data.py to fetch the matchday fixtures.")
        else:
            with st.spinner("The models are predicting missing matches..."):
                new_predictions = application.predict_missing(all_fixtures)
                merged_predictions = {
                    model_name: dict(model_predictions)
                    for model_name, model_predictions in disk_predictions.items()
                }
                for model_name, model_predictions in new_predictions.items():
                    merged_predictions.setdefault(model_name, {}).update(
                        model_predictions
                    )
                st.session_state.model_predictions = merged_predictions

    if all_fixtures.empty:
        st.info("No stored fixtures found. Run football_data.py to fetch the matchday fixtures.")
    elif fixtures.empty:
        st.info(f"No {fixture_filter.lower()} fixtures found.")

    session_predictions = st.session_state.get("model_predictions", {})
    predictions = {
        model_name: dict(model_predictions)
        for model_name, model_predictions in disk_predictions.items()
    }
    if isinstance(session_predictions, dict):
        for model_name, model_predictions in session_predictions.items():
            if isinstance(model_predictions, dict):
                for match_id, value in model_predictions.items():
                    if not isinstance(value, str):
                        continue
                    value = value.strip()
                    if value and not value.lower().startswith("prediction unavailable"):
                        predictions.setdefault(model_name, {})[match_id] = value
    st.session_state.model_predictions = predictions

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
        match_id = str(fixture.get("match_id", ""))
        match_predictions = [
            f"{model_name}: {model_predictions[match_id]}"
            for model_name, model_predictions in predictions.items()
            if match_id in model_predictions
        ]
        prediction = " · ".join(match_predictions) or "Prediction pending"
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
    'Sample dashboard · Premier League · Prediction models</div>',
    unsafe_allow_html=True,
)
