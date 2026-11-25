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

stylesheet = Path(__file__).with_name("style.css").read_text(encoding="utf-8")
st.markdown(f"<style>{stylesheet}</style>", unsafe_allow_html=True)

def _set_active_page(page: str) -> None:
    st.session_state.active_page = page


page = st.session_state.get("active_page", "Matchday")
navigation = st.columns([1.5, 1.1, 1.1, 1.3])
with navigation[0]:
    st.markdown('<div class="nav-brand">TOUCHLINE <span>AI</span></div>', unsafe_allow_html=True)
with navigation[1]:
    st.button(
        "Matchday",
        type="primary" if page == "Matchday" else "secondary",
        use_container_width=True,
        key="nav_matchday",
        on_click=_set_active_page,
        args=("Matchday",),
    )
with navigation[2]:
    st.button(
        "Leaderboard",
        type="primary" if page == "Leaderboard" else "secondary",
        use_container_width=True,
        key="nav_leaderboard",
        on_click=_set_active_page,
        args=("Leaderboard",),
    )
with navigation[3]:
    st.markdown(
        '<div class="nav-league"><div class="league-pill">PREMIER LEAGUE</div></div>',
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

st.markdown('<div class="section-head"><p class="section-title">Prediction models</p></div>', unsafe_allow_html=True)
if model_summaries:
    model_rows = [
        {
            "Model": model["name"],
            "Provider": model["model_id"],
            "Predictions": int(model["prediction_count"]),
            "Points": int(model["points"]),
        }
        for model in model_summaries
    ]
    st.dataframe(model_rows, use_container_width=True, hide_index=True)
else:
    st.info("No prediction providers registered.")

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
            f'<div class="teams">{home} <span class="team-separator">vs</span> {away}</div>'
            f'<div class="score">{score}</div>'
            f'<div class="prediction">{prediction}</div>'
            f'<div class="status">{status}</div>'
            '</div>',
            unsafe_allow_html=True,
        )

st.markdown(
    '<div class="dashboard-footer">'
    'Sample dashboard · Premier League · Prediction models</div>',
    unsafe_allow_html=True,
)
