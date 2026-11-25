"""UI-only prototype for the AI Football Premier League dashboard."""

import html
from pathlib import Path

import altair as alt
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
    .block-container { max-width: 1220px; padding-top: 1.5rem; padding-bottom: 3rem; }
    .brand { font-weight:800; font-size:1.05rem; letter-spacing:-.04em; color:#102b23; }
    .brand span { color:#247b53; }
    .league-pill { background:#e3f5df; color:#267443; border-radius:99px; padding:.42rem .75rem; font-size:.72rem; font-weight:700; letter-spacing:.08em; }
    .top-nav { display:flex; justify-content:center; align-items:center; gap:1.5rem; border-bottom:1px solid #dce5dc; }
    .top-nav a { position:relative; display:block; padding:.5rem .15rem .65rem; color:#40564a !important; font-size:.9rem; font-weight:700; text-decoration:none; white-space:nowrap; }
    .top-nav a:hover, .top-nav a.active { color:#102b23 !important; }
    .top-nav a.active::after { content:""; position:absolute; right:0; bottom:-1px; left:0; height:3px; background:#176b4c; border-radius:3px 3px 0 0; }
    .top-nav a:focus-visible { outline:3px solid #247b53; outline-offset:3px; border-radius:2px; }
    .eyebrow { text-transform:uppercase; letter-spacing:.16em; font:500 .69rem 'DM Mono', monospace; color:#668071; }
    .hero-title { font-size:clamp(2.25rem, 5vw, 4.1rem); line-height:1.04; font-weight:800; letter-spacing:-.065em; color:#102b23; margin:.6rem 0 .8rem; }
    .hero-copy { color:#718078; font-size:1rem; margin-bottom:1.6rem; }
    .metric { background:white; border:1px solid #e5ebe5; border-radius:14px; padding:1.1rem 1.25rem; min-height:105px; }
    .metric-label { color:#718078; font-size:.75rem; text-transform:uppercase; letter-spacing:.11em; }
    .metric-value { color:#102b23; font-size:1.8rem; font-weight:800; letter-spacing:-.05em; margin-top:.25rem; }
    .section-head { display:flex; justify-content:space-between; align-items:end; margin:2.4rem 0 1rem; }
    .section-title { font-size:1.45rem; letter-spacing:-.045em; font-weight:800; margin:0; color:#102b23; }
    .leaderboard-table-wrap { width:100%; overflow-x:auto; background:#fff; border:1px solid #e5ebe5; border-radius:14px; }
    .leaderboard-table { width:100%; min-width:540px; border-collapse:collapse; font-size:.88rem; }
    .leaderboard-table th, .leaderboard-table td { padding:.85rem 1rem; text-align:left; white-space:nowrap; }
    .leaderboard-table thead th { background:#e3f5df; color:#102b23; font-size:.72rem; font-weight:700; letter-spacing:.08em; text-transform:uppercase; }
    .leaderboard-table tbody th { color:#19372b; font-weight:700; }
    .leaderboard-table tbody td { border-top:1px solid #e5ebe5; color:#19372b; }
    .leaderboard-table .numeric-cell { text-align:right; font-variant-numeric:tabular-nums; }
    .leaderboard-table .points-cell { color:#176b4c; font-weight:800; }
    .leaderboard-table tbody tr:hover th, .leaderboard-table tbody tr:hover td { background:#f5f7f2; }
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

topbar = st.columns([1.2, 1.6, 1])
page = st.query_params.get("page", "Matchday")
if page not in {"Matchday", "Leaderboard"}:
    page = "Matchday"
matchday_active = " active" if page == "Matchday" else ""
matchday_current = ' aria-current="page"' if page == "Matchday" else ""
leaderboard_active = " active" if page == "Leaderboard" else ""
leaderboard_current = ' aria-current="page"' if page == "Leaderboard" else ""

with topbar[0]:
    st.markdown(
        '<div class="brand">TOUCHLINE <span>AI</span></div>',
        unsafe_allow_html=True,
    )
with topbar[1]:
    st.markdown(
        '<nav class="top-nav" aria-label="Workspace">'
        f'<a href="?page=Matchday" target="_self" class="top-nav-link{matchday_active}"'
        f'{matchday_current}>Matchday</a>'
        f'<a href="?page=Leaderboard" target="_self" class="top-nav-link{leaderboard_active}"'
        f'{leaderboard_current}>Leaderboard</a>'
        '</nav>',
        unsafe_allow_html=True,
    )
with topbar[2]:
    st.markdown(
        '<div style="text-align:right"><span class="league-pill">PREMIER LEAGUE</span></div>',
        unsafe_allow_html=True,
    )

st.markdown('<div class="eyebrow">THE BEAUTIFUL GAME, WITH A SECOND OPINION</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-title">Every match has<br>another point of view.</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-copy">Prediction models call the fixtures. The table keeps score.</div>', unsafe_allow_html=True)

toolbar = st.columns([1, 1, 2, 2])
with toolbar[0]:
    season_picker = st.empty()
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
kickoff_dates = pd.to_datetime(all_fixtures["match_date"], errors="coerce")
season_start_years = kickoff_dates.dt.year - (kickoff_dates.dt.month < 8).astype(int)
season_options = sorted(
    {
        f"{int(year)}/{(int(year) + 1) % 100:02d}"
        for year in season_start_years.dropna()
    },
    reverse=True,
)

with season_picker:
    if season_options:
        selected_season = st.selectbox("Season", season_options)
    else:
        st.selectbox("Season", ["No season data"], disabled=True)
        selected_season = None

if selected_season is not None:
    selected_start_year = int(selected_season.split("/", maxsplit=1)[0])
    all_fixtures = all_fixtures.loc[season_start_years == selected_start_year]

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
    model_columns = st.columns(min(3, len(model_summaries)))
    for index, model in enumerate(model_summaries):
        with model_columns[index % len(model_columns)]:
            st.markdown(f"**{model['model_id']}**")
            st.caption(f"Provider: {model['name']}")
else:
    st.caption("No prediction providers registered.")

if page == "Leaderboard":
    st.markdown('<div class="section-head"><p class="section-title">Bot leaderboard</p></div>', unsafe_allow_html=True)
    st.markdown("#### Points over time")
    scores = dashboard["scores"]
    if scores.empty:
        st.info("Points history will appear here after completed matches are scored.")
    else:
        chart_scores = scores.copy()
        chart_scores["match_date"] = (
            pd.to_datetime(chart_scores["match_date"], errors="coerce", utc=True)
            .dt.tz_localize(None)
            .dt.normalize()
        )
        chart_scores["points"] = pd.to_numeric(
            chart_scores["points"], errors="coerce"
        ).fillna(0)
        chart_scores = chart_scores.dropna(subset=["match_date"])
        if selected_season is not None:
            season_start = pd.Timestamp(
                year=selected_start_year, month=8, day=1
            )
            season_end = pd.Timestamp(
                year=selected_start_year + 1, month=8, day=1
            )
            chart_scores = chart_scores.loc[
                (chart_scores["match_date"] >= season_start)
                & (chart_scores["match_date"] < season_end)
            ]

        if chart_scores.empty:
            st.info("No scored matches found for this season.")
        else:
            chart_scores["model"] = chart_scores["model"].map(
                lambda model_id: active_model_names.get(
                    str(model_id), str(model_id)
                )
            )
            daily_points = chart_scores.groupby(
                ["match_date", "model"], as_index=False
            )["points"].sum()
            points_over_time = (
                daily_points.pivot(
                    index="match_date", columns="model", values="points"
                )
                .fillna(0)
                .sort_index()
                .cumsum()
            )
            chart_data = points_over_time.rename_axis("match_date").reset_index().melt(
                id_vars="match_date",
                var_name="model",
                value_name="points",
            )
            points_chart = (
                alt.Chart(chart_data)
                .mark_line(point=True)
                .encode(
                    x=alt.X("match_date:T", title="Match date"),
                    y=alt.Y(
                        "points:Q",
                        title="Cumulative points",
                        scale=alt.Scale(domainMin=0),
                    ),
                    color=alt.Color(
                        "model:N",
                        title="Model",
                        scale=alt.Scale(
                            range=["#176b4c", "#247b53", "#8fbe9a", "#102b23"]
                        ),
                    ),
                    tooltip=[
                        alt.Tooltip("match_date:T", title="Match date"),
                        alt.Tooltip("model:N", title="Model"),
                        alt.Tooltip("points:Q", title="Cumulative points"),
                    ],
                )
                .properties(
                    title="Cumulative Model Points by Match Date",
                    height=350,
                    background="#ffffff",
                )
                .configure_axis(
                    domainColor="#dce5dc",
                    gridColor="#e5ebe5",
                    labelColor="#718078",
                    labelFont="DM Sans",
                    tickColor="#dce5dc",
                    titleColor="#668071",
                    titleFont="DM Sans",
                )
                .configure_legend(
                    labelColor="#40564a",
                    labelFont="DM Sans",
                    titleColor="#102b23",
                    titleFont="DM Sans",
                )
                .configure_title(
                    anchor="start",
                    color="#102b23",
                    font="DM Sans",
                    fontSize=16,
                )
                .configure_view(stroke="#e5ebe5")
            )
            st.altair_chart(points_chart, use_container_width=True)

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
        columns = ["Model", "Predictions", "Correct", "Points", "Accuracy"]
        header_cells = "".join(
            f'<th scope="col" class="{"numeric-cell" if column != "Model" else ""}">'
            f"{html.escape(column)}</th>"
            for column in columns
        )
        table_rows = []
        for row in display_rows:
            model_cell = (
                f'<th scope="row">{html.escape(str(row["Model"]))}</th>'
            )
            value_cells = []
            for column in columns[1:]:
                classes = "numeric-cell"
                if column == "Points":
                    classes += " points-cell"
                value_cells.append(
                    f'<td class="{classes}">'
                    f"{html.escape(str(row[column]))}</td>"
                )
            table_rows.append(
                f"<tr>{model_cell}{''.join(value_cells)}</tr>"
            )
        st.markdown(
            '<div class="leaderboard-table-wrap">'
            '<table class="leaderboard-table">'
            f"<thead><tr>{header_cells}</tr></thead>"
            f"<tbody>{''.join(table_rows)}</tbody>"
            "</table></div>",
            unsafe_allow_html=True,
        )
else:
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
