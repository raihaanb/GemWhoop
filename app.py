"""GemWhoop — a Garmin-powered personal performance dashboard.

Run with:
    streamlit run improved_gemwhoop_dashboard.py

Credentials can be supplied through Streamlit secrets or environment variables:
    GARMIN_EMAIL, GARMIN_PASSWORD, GEMINI_API_KEY, GEMINI_MODEL

Never commit real credentials to source control. Garmin data access relies on the
unofficial `garminconnect` Python library and the endpoints enabled for your account.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import os
from collections.abc import Mapping
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    from garminconnect import Garmin
except ImportError:  # Allow the app to display a useful dependency error.
    Garmin = None  # type: ignore[assignment,misc]

try:
    from google import genai
except ImportError:
    genai = None  # type: ignore[assignment,misc]


# -----------------------------------------------------------------------------
# Page setup and visual system
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="GemWhoop | Personal Performance",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="expanded",
)

ACCENT = "#D6FF3F"
PURPLE = "#B7A1FF"
BLUE = "#78A9FF"
ORANGE = "#FFB86B"
RED = "#FF7373"
GREEN = "#72E2B8"
MUTED = "#8A91A1"
BG = "#08090B"
PANEL = "#12151B"
BORDER = "#242832"

st.markdown(
    f"""
    <style>
    :root {{ color-scheme: dark; }}
    .stApp {{ background: {BG}; color: #F3F4F6; }}
    [data-testid="stHeader"] {{ background: rgba(8,9,11,0.84); }}
    [data-testid="stSidebar"] {{ background: #0D0F13; border-right: 1px solid {BORDER}; }}
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] label {{ color: #C4C8D1; }}
    h1, h2, h3 {{ letter-spacing: -0.035em; color: #F8F9FB; }}
    h1 {{ font-size: clamp(1.65rem, 4vw, 2.65rem) !important; font-weight: 800 !important; }}
    h2 {{ font-size: clamp(1.2rem, 2.5vw, 1.65rem) !important; font-weight: 750 !important; }}
    h3 {{ font-size: 1.08rem !important; font-weight: 700 !important; }}
    p, li, label {{ line-height: 1.55; }}
    [data-testid="stMetric"] {{ background: {PANEL}; border: 1px solid {BORDER}; padding: 14px 16px; border-radius: 16px; }}
    [data-testid="stMetricLabel"] {{ color: {MUTED}; }}
    [data-testid="stMetricValue"] {{ color: #F8F9FB; font-weight: 750; }}
    [data-testid="stTabs"] button {{ border-radius: 999px; padding: 0.55rem 0.9rem; }}
    [data-testid="stTabs"] button[aria-selected="true"] {{ background: #252A31; color: {ACCENT}; }}
    div.stButton > button {{ border-radius: 999px; min-height: 2.55rem; border: 1px solid #343944; background: #1B1F26; color: #F4F5F7; font-weight: 650; transition: all 140ms ease; }}
    div.stButton > button:hover {{ border-color: {ACCENT}; color: {ACCENT}; transform: translateY(-1px); }}
    div.stButton > button[kind="primary"] {{ background: {ACCENT}; color: #10120A; border-color: {ACCENT}; }}
    div[data-testid="stExpander"] {{ border: 1px solid {BORDER}; border-radius: 14px; background: #0D1015; }}
    div[data-testid="stAlert"] {{ border-radius: 14px; }}
    [data-testid="stDataFrame"] {{ border: 1px solid {BORDER}; border-radius: 12px; overflow: hidden; }}
    .hero-row {{ display:flex; align-items:center; justify-content:space-between; gap:1rem; flex-wrap:wrap; margin-bottom:0.35rem; }}
    .eyebrow {{ color:{ACCENT}; font-size:0.72rem; font-weight:800; text-transform:uppercase; letter-spacing:0.18em; margin-bottom:0.45rem; }}
    .subtle {{ color:{MUTED}; font-size:0.88rem; }}
    .section-label {{ text-transform:uppercase; color:{MUTED}; font-size:0.72rem; font-weight:800; letter-spacing:0.15em; margin-bottom:0.35rem; }}
    .metric-card {{ background:linear-gradient(145deg,#171B22,#0F1116); border:1px solid {BORDER}; border-radius:22px; padding:18px 12px 14px; min-height:245px; text-align:center; box-shadow:0 10px 28px rgba(0,0,0,0.14); }}
    .gauge {{ width:126px; height:126px; border-radius:50%; margin:14px auto 12px; padding:9px; display:flex; align-items:center; justify-content:center; }}
    .gauge-hole {{ width:100%; height:100%; background:#101319; border-radius:50%; display:flex; align-items:center; justify-content:center; flex-direction:column; border:1px solid #282D36; }}
    .gauge-value {{ font-size:2rem; line-height:1.05; font-weight:850; color:#F7F8FA; letter-spacing:-0.05em; }}
    .gauge-unit {{ color:{MUTED}; font-size:0.73rem; margin-top:4px; }}
    .card-label {{ font-size:0.78rem; font-weight:800; text-transform:uppercase; letter-spacing:0.12em; color:#B5BBC6; }}
    .card-caption {{ font-size:0.78rem; color:{MUTED}; min-height:1.3rem; margin-bottom:0.8rem; }}
    .no-data {{ display:inline-block; padding:3px 9px; border-radius:999px; border:1px solid #373C45; color:#A8ADB7; background:#181B21; font-size:0.73rem; font-weight:700; }}
    .coach-panel {{ background:linear-gradient(140deg,#161B20,#101216); border:1px solid #2A3038; border-radius:20px; padding:20px; }}
    .tiny-dot {{ display:inline-block; width:7px; height:7px; border-radius:50%; background:{GREEN}; margin-right:8px; }}
    hr {{ border-color:{BORDER}; }}
    @media (max-width: 700px) {{
      .metric-card {{ min-height:220px; padding:14px 8px 12px; }}
      .gauge {{ width:106px; height:106px; }}
      .gauge-value {{ font-size:1.7rem; }}
      [data-testid="stMetric"] {{ padding:10px 11px; }}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Secrets and common data helpers
# -----------------------------------------------------------------------------
def get_config_value(name: str, default: str = "") -> str:
    """Read a setting from Streamlit secrets first, then the environment."""
    try:
        value = st.secrets.get(name, None)
    except Exception:
        value = None
    if value is not None and str(value).strip():
        return str(value).strip()
    return os.getenv(name, default).strip()


def to_plain(value: Any) -> Any:
    """Convert Garmin SDK responses / Pydantic models into JSON-safe values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (dt.date, dt.datetime)):
        return value.isoformat()
    if hasattr(value, "model_dump"):
        try:
            return to_plain(value.model_dump(mode="json", exclude_none=False))
        except Exception:
            try:
                return to_plain(value.model_dump(exclude_none=False))
            except Exception:
                pass
    if hasattr(value, "dict") and callable(value.dict):
        try:
            return to_plain(value.dict())
        except Exception:
            pass
    if isinstance(value, Mapping):
        return {str(k): to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_plain(v) for v in value]
    if hasattr(value, "__dict__"):
        try:
            return to_plain(vars(value))
        except Exception:
            pass
    return str(value)


def is_present(value: Any) -> bool:
    """An explicit zero is data; None and empty containers are not."""
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip() != ""
    if isinstance(value, (dict, list, tuple, set)):
        return len(value) > 0
    try:
        return not bool(pd.isna(value))
    except (TypeError, ValueError):
        return True


def parse_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        if math.isfinite(float(value)):
            return float(value)
        return None
    if isinstance(value, str):
        cleaned = value.strip().replace(",", "")
        try:
            num = float(cleaned)
            return num if math.isfinite(num) else None
        except ValueError:
            return None
    return None


def get_path(payload: Any, path: str) -> Any:
    current = payload
    for part in path.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        elif isinstance(current, list) and part.isdigit() and int(part) < len(current):
            current = current[int(part)]
        else:
            return None
    return current


def first_value(payloads: list[tuple[Any, list[str]]]) -> Any:
    """Return the first explicitly present value from ordered path alternatives."""
    for payload, paths in payloads:
        for path in paths:
            value = get_path(payload, path)
            if is_present(value):
                return value
    return None


def find_numeric_by_key(payload: Any, target_keys: set[str]) -> float | None:
    """Case-insensitive recursive fallback for renamed fields in Garmin payloads."""
    if isinstance(payload, dict):
        # Exact-key matches are checked before traversing nested children.
        for key, value in payload.items():
            if str(key).lower() in target_keys:
                number = parse_number(value)
                if number is not None:
                    return number
        for value in payload.values():
            found = find_numeric_by_key(value, target_keys)
            if found is not None:
                return found
    elif isinstance(payload, list):
        for value in payload:
            found = find_numeric_by_key(value, target_keys)
            if found is not None:
                return found
    return None


def fmt_number(value: Any, decimals: int = 0, suffix: str = "") -> str:
    number = parse_number(value)
    if number is None:
        return "No data"
    rendered = f"{number:,.{decimals}f}" if decimals else f"{number:,.0f}"
    return f"{rendered}{suffix}"


def flatten_payload(source: str, payload: Any) -> list[dict[str, str]]:
    """Flatten every returned leaf for browsing; the raw JSON remains available too."""
    rows: list[dict[str, str]] = []

    def walk(value: Any, path: str) -> None:
        if isinstance(value, dict):
            if not value:
                rows.append({"Source": source, "Metric / path": path or "(root)", "Value": "{}"})
            for key, nested in value.items():
                next_path = f"{path}.{key}" if path else str(key)
                walk(nested, next_path)
        elif isinstance(value, list):
            if not value:
                rows.append({"Source": source, "Metric / path": path, "Value": "[]"})
            elif all(isinstance(item, dict) for item in value):
                for index, item in enumerate(value):
                    walk(item, f"{path}[{index}]")
            else:
                rows.append({
                    "Source": source,
                    "Metric / path": path,
                    "Value": json.dumps(value, ensure_ascii=False, default=str),
                })
        else:
            rows.append({"Source": source, "Metric / path": path, "Value": json.dumps(value, ensure_ascii=False, default=str)})

    if is_present(payload):
        walk(payload, "")
    return rows


def safe_api_call(api: Any, method_name: str, *args: Any) -> tuple[Any, str | None]:
    """Call one optional endpoint without letting a single failure stop the app."""
    method = getattr(api, method_name, None)
    if not callable(method):
        return None, "Endpoint not available in the installed garminconnect version."
    try:
        result = to_plain(method(*args))
        return result, None
    except Exception as exc:  # Garmin accounts/devices expose different subsets.
        message = str(exc).strip().replace("\n", " ")
        if len(message) > 240:
            message = message[:237] + "..."
        return None, f"{type(exc).__name__}: {message or 'request failed'}"


# -----------------------------------------------------------------------------
# Garmin data collection
# -----------------------------------------------------------------------------
def fetch_day_bundle(api: Any, date_str: str, include_extended: bool) -> tuple[dict[str, Any], dict[str, str]]:
    payloads: dict[str, Any] = {}
    errors: dict[str, str] = {}

    # Core sources power the headline metrics and 7-day charts.
    core_calls = {
        "daily_stats": ("get_stats", (date_str,)),
        "sleep": ("get_sleep_data", (date_str,)),
        "hrv": ("get_hrv_data", (date_str,)),
        "training_readiness": ("get_training_readiness", (date_str,)),
    }
    for key, (method, args) in core_calls.items():
        result, error = safe_api_call(api, method, *args)
        payloads[key] = result
        if error:
            errors[key] = error

    if include_extended:
        # Broad, read-only health/performance telemetry. Unsupported methods and
        # unavailable account/device data are recorded rather than fabricated.
        extended_calls: dict[str, tuple[str, tuple[Any, ...]]] = {
            "training_readiness_morning": ("get_morning_training_readiness", (date_str,)),
            "training_status": ("get_training_status", (date_str,)),
            "training_load_balance_4wk": ("get_training_four_week_load_balance", (date_str,)),
            "max_metrics_vo2": ("get_max_metrics", (date_str,)),
            "all_day_stress": ("get_all_day_stress", (date_str,)),
            "respiration": ("get_respiration_data", (date_str,)),
            "spo2": ("get_spo2_data", (date_str,)),
            "hydration": ("get_hydration_all_data", (date_str,)),
            "intensity_minutes": ("get_intensity_minutes_data", (date_str,)),
            "body_battery_events": ("get_body_battery_events", (date_str,)),
            "naps": ("get_naps", (date_str,)),
            "heart_rate_samples": ("get_heart_rates", (date_str,)),
            "daily_movement": ("get_daily_movement", (date_str,)),
            "floors": ("get_floors", (date_str,)),
            "activities": ("get_activities_by_date", (date_str, date_str)),
            "steps_range": ("get_daily_steps", (date_str, date_str)),
            "hr_daily_stats": ("get_daily_heart_rate_stats", (date_str, date_str)),
            "respiration_daily_stats": ("get_daily_respiration_stats", (date_str, date_str)),
            "stress_daily_stats": ("get_daily_stress_stats", (date_str, date_str)),
            "intensity_minutes_daily_stats": ("get_daily_intensity_minutes_stats", (date_str, date_str)),
            "floors_daily_stats": ("get_daily_floors_stats", (date_str, date_str)),
            "body_battery_daily_stats": ("get_daily_body_battery_stats", (date_str, date_str)),
            "body_battery_series": ("get_body_battery", (date_str, date_str)),
            "daily_summaries": ("get_daily_summaries", (date_str, date_str)),
            "sleep_daily_summary": ("get_sleep_daily", (date_str, date_str)),
            "all_day_events": ("get_all_day_events", (date_str,)),
            "blood_pressure": ("get_blood_pressure_day", (date_str,)),
            "daily_weigh_ins": ("get_daily_weigh_ins", (date_str,)),
            "body_composition": ("get_body_composition", (date_str, date_str)),
            "glucose": ("get_glucose_data", (date_str,)),
            "lifestyle_log": ("get_lifestyle_logging_data", (date_str,)),
            "endurance_score": ("get_endurance_score", (date_str, date_str)),
            "hill_score": ("get_hill_score", (date_str, date_str)),
            "running_economy": ("get_running_economy", (date_str,)),
            "heat_altitude_acclimation": ("get_heat_altitude_acclimation", (date_str,)),
            "fitness_age_details": ("get_fitness_age_details", (date_str,)),
            "health_snapshots": ("get_health_snapshots", (date_str,)),
            "nutrition_food_log": ("get_nutrition_daily_food_log", (date_str,)),
        }
        for key, (method, args) in extended_calls.items():
            result, error = safe_api_call(api, method, *args)
            payloads[key] = result
            if error:
                errors[key] = error

        # Activity summaries often omit splits, cadence, power, and other detailed
        # metrics. Fetch each available activity detail record for the selected day.
        activities = payloads.get("activities")
        if isinstance(activities, list):
            for index, activity in enumerate(activities):
                if not isinstance(activity, dict):
                    continue
                activity_id = activity.get("activityId") or activity.get("activity_id") or activity.get("id")
                if activity_id is None:
                    continue
                detail_key = f"activity_detail_{activity_id}"
                detail_result, detail_error = safe_api_call(api, "get_activity_details", activity_id)
                payloads[detail_key] = detail_result
                if detail_error:
                    errors[detail_key] = detail_error

    return payloads, errors


def build_history(api: Any, selected_date: dt.date, current_payloads: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, str]]:
    rows: list[dict[str, Any]] = []
    errors: dict[str, str] = {}
    dates = [selected_date - dt.timedelta(days=offset) for offset in range(6, -1, -1)]

    for day in dates:
        day_str = day.isoformat()
        if day == selected_date:
            payloads = current_payloads
        else:
            payloads = {}
            for key, method in (
                ("daily_stats", "get_stats"),
                ("sleep", "get_sleep_data"),
                ("hrv", "get_hrv_data"),
                ("training_readiness", "get_training_readiness"),
            ):
                result, error = safe_api_call(api, method, day_str)
                payloads[key] = result
                if error:
                    errors[f"{day_str}:{key}"] = error

        stats = payloads.get("daily_stats")
        sleep = payloads.get("sleep")
        hrv = payloads.get("hrv")
        readiness = payloads.get("training_readiness")

        body_battery = first_value([
            (stats, ["bodyBatteryMostRecentValue", "bodyBatteryValue", "bodyBattery"]),
            (payloads.get("body_battery_series"), ["bodyBatteryMostRecentValue", "bodyBatteryValue", "bodyBattery"]),
        ])
        sleep_score = extract_sleep_score(sleep)
        readiness_score = extract_readiness_score(readiness)
        rhr = first_value([(stats, ["restingHeartRate", "restingHr"]), (stats, ["restingHeartRateValue"])])
        hrv_value = extract_hrv_value(hrv, stats)

        rows.append({
            "Date": day_str,
            "Body Battery": parse_number(body_battery),
            "Sleep Score": parse_number(sleep_score),
            "Training Readiness": parse_number(readiness_score),
            "Resting HR (bpm)": parse_number(rhr),
            "HRV (ms)": parse_number(hrv_value),
        })

    return pd.DataFrame(rows), errors


def extract_sleep_score(sleep: Any) -> float | None:
    value = first_value([
        (sleep, ["dailySleepDTO.sleepScores.overall.value"]),
        (sleep, ["dailySleepDTO.sleepScorePercentage"]),
        (sleep, ["dailySleepDTO.sleepScore"]),
        (sleep, ["sleepScores.overall.value"]),
        (sleep, ["sleepScorePercentage"]),
        (sleep, ["sleepScore"]),
    ])
    number = parse_number(value)
    return number if number is not None else find_numeric_by_key(sleep, {"sleepscorepercentage", "sleepscore"})


def extract_readiness_score(readiness: Any) -> float | None:
    number = first_value([
        (readiness, ["score", "trainingReadinessScore", "overallScore", "readinessScore", "value"]),
        (readiness, ["trainingReadiness.score", "trainingReadinessScore.value"]),
    ])
    parsed = parse_number(number)
    if parsed is not None:
        return parsed
    return find_numeric_by_key(readiness, {"trainingreadinessscore", "readinessscore", "overallscore", "score"})


def extract_hrv_value(hrv: Any, stats: Any = None) -> float | None:
    value = first_value([
        (hrv, ["hrvSummary.lastNightAvg", "hrvSummary.lastNightAverage", "lastNightAvg", "lastNightAverage"]),
        (hrv, ["hrvSummary.weeklyAvg", "hrvSummary.weeklyAverage", "weeklyAvg", "weeklyAverage"]),
        (stats, ["lastNightAvgHrv", "hrvStatus.weeklyAverage"]),
    ])
    parsed = parse_number(value)
    if parsed is not None:
        return parsed
    found = find_numeric_by_key(hrv, {"lastnightavghrv", "lastnightavg", "weeklyaverage", "weeklyavg"})
    if found is not None:
        return found
    return find_numeric_by_key(stats, {"lastnightavghrv"})


def duration_hours(payload: Any, path_options: list[str]) -> float | None:
    raw = first_value([(payload, path_options)])
    seconds = parse_number(raw)
    return seconds / 3600.0 if seconds is not None else None


def daily_metric_context(date_str: str, payloads: dict[str, Any], history: pd.DataFrame) -> dict[str, Any]:
    stats = payloads.get("daily_stats")
    sleep = payloads.get("sleep")
    readiness = payloads.get("training_readiness")
    battery = first_value([
        (stats, ["bodyBatteryMostRecentValue", "bodyBatteryValue", "bodyBattery"]),
        (payloads.get("body_battery_series"), ["bodyBatteryMostRecentValue", "bodyBatteryValue", "bodyBattery"]),
    ])
    sleep_score = extract_sleep_score(sleep)
    readiness_score = extract_readiness_score(readiness)
    rhr = first_value([(stats, ["restingHeartRate", "restingHr", "restingHeartRateValue"])])
    hrv = extract_hrv_value(payloads.get("hrv"), stats)
    deep_hours = duration_hours(sleep, [
        "dailySleepDTO.deepSleepSeconds", "deepSleepSeconds", "sleep.deepSleepSeconds"
    ])
    rem_hours = duration_hours(sleep, [
        "dailySleepDTO.remSleepSeconds", "remSleepSeconds", "sleep.remSleepSeconds"
    ])
    total_hours = duration_hours(sleep, [
        "dailySleepDTO.sleepTimeSeconds", "sleepTimeSeconds", "sleep.sleepTimeSeconds"
    ])
    return {
        "date": date_str,
        "goals": {},  # The coach adds user goals immediately before each request.
        "headline": {
            "body_battery": battery,
            "sleep_score": sleep_score,
            "training_readiness": readiness_score,
            "resting_heart_rate_bpm": rhr,
            "hrv_ms": hrv,
            "sleep_hours": total_hours,
            "deep_sleep_hours": deep_hours,
            "rem_sleep_hours": rem_hours,
        },
        "seven_day_history": history.to_dict(orient="records"),
        "available_garmin_sources": [key for key, value in payloads.items() if is_present(value)],
        "missing_sources": [key for key, value in payloads.items() if not is_present(value)],
        "raw_day_payloads": payloads,
    }


# -----------------------------------------------------------------------------
# Chart / component helpers
# -----------------------------------------------------------------------------
def render_gauge(title: str, value: Any, max_value: float, unit: str, color: str, caption: str, key: str) -> None:
    number = parse_number(value)
    if number is None:
        ratio = 0
        value_text = "—"
        unit_text = "NO DATA"
        caption_text = "No reading for this date"
    else:
        ratio = max(0.0, min(100.0, number / max_value * 100.0)) if max_value > 0 else 0.0
        value_text = f"{number:.0f}" if float(number).is_integer() else f"{number:.1f}"
        unit_text = unit.upper() if unit else ""
        caption_text = caption

    html = f"""
    <div class="metric-card">
      <div class="card-label">{title}</div>
      <div class="gauge" style="background:conic-gradient({color} {ratio:.2f}%, #292E37 0)">
        <div class="gauge-hole">
          <div class="gauge-value">{value_text}</div>
          <div class="gauge-unit">{unit_text}</div>
        </div>
      </div>
      <div class="card-caption">{caption_text}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
    st.button("View details ↗", key=f"view_{key}", use_container_width=True, on_click=set_selected_detail, args=(key,))


def set_selected_detail(key: str) -> None:
    st.session_state["selected_detail"] = key


def make_line_chart(df: pd.DataFrame, columns: list[tuple[str, str]]) -> go.Figure:
    fig = go.Figure()
    palette = [ACCENT, PURPLE, BLUE, ORANGE, GREEN]
    for index, (column, label) in enumerate(columns):
        if column not in df.columns:
            continue
        values = [parse_number(x) for x in df[column].tolist()]
        if not any(value is not None for value in values):
            continue
        fig.add_trace(go.Scatter(
            x=df["Date"], y=values, mode="lines+markers", name=label,
            connectgaps=False, line={"color": palette[index % len(palette)], "width": 2.5},
            marker={"size": 6},
            hovertemplate="%{x}<br>" + label + ": %{y}<extra></extra>",
        ))
    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=PANEL,
        font={"color": "#D7DAE0", "family": "Inter, system-ui, sans-serif", "size": 12},
        xaxis={"gridcolor": BORDER, "showgrid": False, "title": None},
        yaxis={"gridcolor": BORDER, "zerolinecolor": BORDER, "title": None},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        margin={"l": 8, "r": 8, "t": 28, "b": 8},
        height=310,
        hovermode="x unified",
    )
    return fig


def section_payload_table(payloads: dict[str, Any], source_keys: list[str], height: int = 420) -> None:
    rows: list[dict[str, str]] = []
    for key in source_keys:
        rows.extend(flatten_payload(key, payloads.get(key)))
    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, hide_index=True, use_container_width=True, height=height)
    else:
        st.markdown('<span class="no-data">No data returned for this section</span>', unsafe_allow_html=True)


def render_source_status(payloads: dict[str, Any], errors: dict[str, str]) -> None:
    total = len(payloads)
    available = sum(1 for value in payloads.values() if is_present(value))
    st.caption(f"{available} of {total} requested data sources returned content. Empty/missing sources remain unavailable — no fallback scores are generated.")
    with st.expander(f"Data-source diagnostics ({len(errors)} notices)", expanded=False):
        if errors:
            for key, message in errors.items():
                st.write(f"**{key}** — {message}")
        else:
            st.write("No endpoint errors reported for the selected date.")


def safe_json_download(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str).encode("utf-8")


def call_gemini(api_key: str, model: str, prompt: str) -> str:
    if genai is None:
        raise RuntimeError("The google-genai package is not installed. Install it from requirements.txt.")
    if not api_key:
        raise RuntimeError("Add a Gemini API key in Streamlit secrets, an environment variable, or the sidebar.")
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(model=model, contents=prompt)
    result = getattr(response, "text", None)
    if not result or not str(result).strip():
        raise RuntimeError("Gemini returned an empty response. Try again or choose a different model ID.")
    return str(result).strip()


def compact_for_coach(value: Any, max_list_items: int = 8) -> Any:
    """Keep day-level fields while summarising high-frequency samples for the LLM."""
    if isinstance(value, dict):
        return {str(key): compact_for_coach(item, max_list_items) for key, item in value.items()}
    if isinstance(value, list):
        if len(value) > max_list_items:
            return {"omitted_sample_series": True, "item_count": len(value), "note": "Full samples remain available in the Garmin data explorer."}
        return [compact_for_coach(item, max_list_items) for item in value]
    if isinstance(value, str) and len(value) > 600:
        return value[:600] + "… [text shortened for coaching context; full payload remains available]"
    return value


def coach_prompt(goals: dict[str, str], context: dict[str, Any], question: str | None = None, chat_history: list[dict[str, str]] | None = None) -> str:
    history = chat_history or []
    conversation = "\n".join(
        f"{item.get('role', 'user').upper()}: {item.get('content', '')}"
        for item in history[-8:]
    )
    full_payloads = context.get("raw_day_payloads", {})
    compact_sources = {
        key: compact_for_coach(value)
        for key, value in full_payloads.items()
        if key not in {"heart_rate_samples", "daily_movement", "activities", "body_battery_events"}
        and is_present(value)
    }
    coach_data = {
        "date": context.get("date"),
        "headline_measurements": context.get("headline", {}),
        "seven_day_history": context.get("seven_day_history", []),
        "available_sources": context.get("available_garmin_sources", []),
        "daily_metric_payloads": compact_sources,
    }
    return f"""You are a focused, evidence-led personal performance coach. Be concise and direct.

USER GOALS (must guide every recommendation):
- Near-term: {goals.get('short', '').strip() or 'Not provided'}
- Longer-term: {goals.get('long', '').strip() or 'Not provided'}

GARMIN DATA FOR {context.get('date')} (JSON; null means unavailable, not zero):
{json.dumps(coach_data, ensure_ascii=False, default=str)}

RULES:
- Use only measurements present in the data. Never invent, impute, or substitute stale readings.
- Explicitly say when a key metric is missing; do not treat missing as a bad or good score.
- Cite the specific number/date/trend behind each claim, and relate it to the stated goals.
- No generic advice, boilerplate, filler, or automatic hydration/sleep recommendations without supporting evidence.
- Make one to three concrete actions, with measurable thresholds or timing where supported by the data.
- Distinguish observed facts from interpretation. Do not promise a 48-hour forecast or state certainty the data cannot support.
- Do not diagnose medical conditions. If data suggest a potentially urgent health concern, advise appropriate clinical care without diagnosing.
- Keep the response to 120 words unless the user explicitly asks for more detail.

Recent conversation:
{conversation or '(No previous chat messages)'}

User request:
{question or 'Give me a daily coaching check-in tied to both goals. Lead with the single most important finding, then actions.'}
"""


@st.cache_resource(show_spinner=False)
def _get_garmin_client(email: str, password: str) -> Any:
    if Garmin is None:
        raise RuntimeError("The garminconnect package is not installed. Install requirements.txt.")
    client = Garmin(email, password)
    client.login()
    return client


# -----------------------------------------------------------------------------
# Sidebar: configuration, date and goals
# -----------------------------------------------------------------------------
st.sidebar.markdown('<div class="eyebrow">GEMWHOOP / CONTROL ROOM</div>', unsafe_allow_html=True)
st.sidebar.title("Settings")
selected_date = st.sidebar.date_input("Analysis date", value=dt.date.today(), max_value=dt.date.today())
include_extended = st.sidebar.toggle("Fetch extended Garmin telemetry", value=True, help="Requests additional health, activity, training and device endpoints. Some may not be supported by your model or account.")
st.sidebar.caption("Extended telemetry provides broader coverage but makes more Garmin API requests.")

stored_email = get_config_value("GARMIN_EMAIL")
stored_password = get_config_value("GARMIN_PASSWORD")
stored_gemini_key = get_config_value("GEMINI_API_KEY")
model_name = get_config_value("GEMINI_MODEL", "gemini-3.8-flash")

with st.sidebar.expander("Connections & credentials", expanded=not bool(stored_email and stored_password)):
    if Garmin is None:
        st.error("`garminconnect` is not installed. Install the packages in requirements.txt.")
    garmin_email = st.text_input("Garmin email", value=stored_email, key="garmin_email_input")
    garmin_password = st.text_input("Garmin password", value=stored_password, type="password", key="garmin_password_input")
    gemini_api_key = st.text_input("Gemini API key", value=stored_gemini_key, type="password", key="gemini_api_key_input")
    model_name = st.text_input("Gemini model ID", value=model_name, key="gemini_model_input")
    st.caption("For regular use, configure these in .streamlit/secrets.toml or environment variables rather than typing them each session.")

refresh = st.sidebar.button("↻ Refresh Garmin data", use_container_width=True)
if refresh:
    st.session_state["refresh_id"] = int(st.session_state.get("refresh_id", 0)) + 1
    st.session_state.pop("loaded_bundle", None)
    try:
        if Garmin is not None:
            # Drop any cached authentication client so refreshed credentials/tokens are used.
            _get_garmin_client.clear()
    except Exception:
        pass

st.sidebar.divider()
st.sidebar.markdown("**Goal setting lives in the Coach tab.** The coach stays inactive until you enter your goals.")


# -----------------------------------------------------------------------------
# Authentication and selected-day bundle loading
# -----------------------------------------------------------------------------
def load_data(api: Any, day: dt.date, extended: bool) -> tuple[dict[str, Any], dict[str, str], pd.DataFrame, dict[str, str]]:
    day_str = day.isoformat()
    current_payloads, endpoint_errors = fetch_day_bundle(api, day_str, extended)
    history_df, history_errors = build_history(api, day, current_payloads)
    return current_payloads, endpoint_errors, history_df, history_errors


# -----------------------------------------------------------------------------
# Page header and connection
# -----------------------------------------------------------------------------
st.markdown('<div class="eyebrow">PERSONAL HEALTH / PERFORMANCE</div>', unsafe_allow_html=True)
st.title("Your performance, without the noise.")
st.markdown(
    f'<div class="subtle"><span class="tiny-dot"></span>GARMIN CONNECT &nbsp;·&nbsp; SELECTED DATE: {selected_date.strftime("%a, %d %b %Y")} &nbsp;·&nbsp; GOAL-LED GEMINI COACHING</div>',
    unsafe_allow_html=True,
)
st.write("")

if not garmin_email or not garmin_password:
    st.warning("Connect your Garmin account to load your personal metrics. Add your details in **Connections & credentials** in the sidebar or configure Streamlit secrets.")
    st.stop()

loaded_key = (garmin_email, selected_date.isoformat(), bool(include_extended), int(st.session_state.get("refresh_id", 0)))
loaded_bundle = st.session_state.get("loaded_bundle")
if not loaded_bundle or loaded_bundle.get("key") != loaded_key:
    try:
        with st.spinner("Connecting to Garmin and loading available measurements…"):
            garmin_api = _get_garmin_client(garmin_email, garmin_password)
            current_payloads, endpoint_errors, history_df, history_errors = load_data(garmin_api, selected_date, include_extended)
        loaded_bundle = {
            "key": loaded_key,
            "payloads": current_payloads,
            "errors": endpoint_errors,
            "history": history_df,
            "history_errors": history_errors,
        }
        st.session_state["loaded_bundle"] = loaded_bundle
    except Exception as exc:
        st.error(f"Could not connect to Garmin or load the selected date: {type(exc).__name__}: {exc}")
        st.info("Check the credentials, Garmin Connect access, MFA/account prompts, and installed garminconnect version. No demo scores are shown when loading fails.")
        st.stop()

payloads: dict[str, Any] = loaded_bundle["payloads"]
endpoint_errors: dict[str, str] = loaded_bundle["errors"]
history_df: pd.DataFrame = loaded_bundle["history"]
history_errors: dict[str, str] = loaded_bundle["history_errors"]
context = daily_metric_context(selected_date.isoformat(), payloads, history_df)

# Extract only values that Garmin actually returned.
stats = payloads.get("daily_stats")
sleep = payloads.get("sleep")
readiness = payloads.get("training_readiness")
body_battery = context["headline"]["body_battery"]
sleep_score = context["headline"]["sleep_score"]
readiness_score = context["headline"]["training_readiness"]
resting_hr = context["headline"]["resting_heart_rate_bpm"]
hrv_value = context["headline"]["hrv_ms"]

if "selected_detail" not in st.session_state:
    st.session_state["selected_detail"] = "body_battery"

# -----------------------------------------------------------------------------
# Main interface
# -----------------------------------------------------------------------------
overview_tab, recovery_tab, sleep_tab, activity_tab, vitals_tab, all_data_tab, coach_tab = st.tabs(
    ["Overview", "Recovery", "Sleep", "Activity", "Vitals", "All Garmin data", "Gemini coach"]
)

with overview_tab:
    st.markdown("## Today at a glance")
    st.caption("Only metrics supplied by Garmin are shown. A dash means the selected date has no available reading for that metric.")
    gauge_cols = st.columns(3, gap="medium")
    with gauge_cols[0]:
        render_gauge("Body Battery", body_battery, 100, "/ 100", ACCENT, "Garmin Body Battery · current recorded value", "body_battery")
    with gauge_cols[1]:
        render_gauge("Training readiness", readiness_score, 100, "/ 100", PURPLE, "Garmin training-readiness score", "training_readiness")
    with gauge_cols[2]:
        render_gauge("Sleep score", sleep_score, 100, "/ 100", BLUE, "Garmin sleep score · selected night", "sleep")

    overview_cols = st.columns(4, gap="small")
    steps_value = first_value([(stats, ["totalSteps", "steps"]), (payloads.get("steps_range"), ["totalSteps", "steps"] )])
    active_calories = first_value([(stats, ["activeKilocalories", "activeCalories"]), (payloads.get("daily_summaries"), ["activeKilocalories", "activeCalories"])])
    total_calories = first_value([(stats, ["totalKilocalories", "totalCalories"]), (payloads.get("daily_summaries"), ["totalKilocalories", "totalCalories"])])
    intensity_minutes = first_value([(stats, ["moderateIntensityMinutes"]), (payloads.get("intensity_minutes"), ["moderateIntensityMinutes"])])
    floors_value = first_value([(stats, ["floorsAscended", "floorsClimbed"]), (payloads.get("floors"), ["floorsAscended", "floorsClimbed"] )])
    quick_metrics = [
        ("Steps", steps_value, ""),
        ("Resting heart rate", resting_hr, " bpm"),
        ("Active calories", active_calories, " kcal"),
        ("Intensity minutes", intensity_minutes, " min"),
    ]
    for col, (label, val, unit) in zip(overview_cols, quick_metrics):
        with col:
            st.metric(label, fmt_number(val, suffix=unit) if is_present(val) else "No data")
    extra_cols = st.columns(4, gap="small")
    with extra_cols[0]:
        st.metric("HRV", fmt_number(hrv_value, decimals=1, suffix=" ms") if is_present(hrv_value) else "No data")
    with extra_cols[1]:
        st.metric("Total calories", fmt_number(total_calories, suffix=" kcal") if is_present(total_calories) else "No data")
    with extra_cols[2]:
        st.metric("Floors climbed", fmt_number(floors_value) if is_present(floors_value) else "No data")
    with extra_cols[3]:
        sleep_hours = context["headline"]["sleep_hours"]
        st.metric("Sleep duration", fmt_number(sleep_hours, decimals=1, suffix=" h") if is_present(sleep_hours) else "No data")

    st.markdown("## Seven-day trend")
    trend_cols = [("Body Battery", "Body Battery"), ("Training Readiness", "Training readiness"), ("Sleep Score", "Sleep score")]
    fig = make_line_chart(history_df, trend_cols)
    if len(fig.data):
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    else:
        st.info("No historical points were returned for the headline metrics in this date range.")
    with st.expander("Historical values", expanded=False):
        history_display = history_df.copy()
        st.dataframe(history_display, hide_index=True, use_container_width=True)
    st.markdown("## Explore a score")
    st.caption("Use a score card’s **View details** action to open its underlying Garmin fields and recent trend.")
    selected_detail = st.session_state.get("selected_detail", "body_battery")
    detail_label = {"body_battery": "Body Battery", "training_readiness": "Training readiness", "sleep": "Sleep"}.get(selected_detail, "Body Battery")
    st.markdown(f"**Selected detail: {detail_label}**")
    if selected_detail == "body_battery":
        st.markdown("**Body Battery source fields**")
        section_payload_table(payloads, ["daily_stats", "body_battery_series", "body_battery_daily_stats", "body_battery_events"])
        st.plotly_chart(make_line_chart(history_df, [("Body Battery", "Body Battery")]), use_container_width=True, config={"displayModeBar": False})
    elif selected_detail == "training_readiness":
        st.markdown("**Training-readiness and load fields**")
        section_payload_table(payloads, ["training_readiness", "training_readiness_morning", "training_status", "training_load_balance_4wk"])
        st.plotly_chart(make_line_chart(history_df, [("Training Readiness", "Training readiness")]), use_container_width=True, config={"displayModeBar": False})
    else:
        st.markdown("**Sleep score, stages, timing and related fields**")
        section_payload_table(payloads, ["sleep", "sleep_daily_summary", "naps"])
        st.plotly_chart(make_line_chart(history_df, [("Sleep Score", "Sleep score")]), use_container_width=True, config={"displayModeBar": False})

with recovery_tab:
    st.markdown("## Recovery & readiness")
    st.plotly_chart(make_line_chart(history_df, [("Body Battery", "Body Battery"), ("Training Readiness", "Training readiness"), ("HRV (ms)", "HRV (ms)"), ("Resting HR (bpm)", "Resting HR (bpm)")]), use_container_width=True, config={"displayModeBar": False})
    st.markdown("### Underlying Garmin measurements")
    section_payload_table(payloads, [
        "daily_stats", "hrv", "training_readiness", "training_readiness_morning", "training_status",
        "training_load_balance_4wk", "all_day_stress", "stress_daily_stats", "body_battery_series",
        "body_battery_daily_stats", "body_battery_events",
    ])

with sleep_tab:
    st.markdown("## Sleep detail")
    sleep_col1, sleep_col2, sleep_col3 = st.columns(3)
    with sleep_col1:
        val = context["headline"]["sleep_hours"]
        st.metric("Total sleep", fmt_number(val, 1, " h") if is_present(val) else "No data")
    with sleep_col2:
        val = context["headline"]["deep_sleep_hours"]
        st.metric("Deep sleep", fmt_number(val, 1, " h") if is_present(val) else "No data")
    with sleep_col3:
        val = context["headline"]["rem_sleep_hours"]
        st.metric("REM sleep", fmt_number(val, 1, " h") if is_present(val) else "No data")
    st.plotly_chart(make_line_chart(history_df, [("Sleep Score", "Sleep score")]), use_container_width=True, config={"displayModeBar": False})
    st.markdown("### All returned sleep fields")
    section_payload_table(payloads, ["sleep", "sleep_daily_summary", "naps"])

with activity_tab:
    st.markdown("## Activity & energy")
    st.caption("Activity data is sourced from the selected date; the full activity records are shown when Garmin returns them.")
    st.markdown("### Daily summary and activity records")
    section_payload_table(payloads, [
        "daily_stats", "activities", "steps_range", "intensity_minutes", "intensity_minutes_daily_stats",
        "floors", "floors_daily_stats", "daily_movement", "daily_summaries",
    ])
    activities = payloads.get("activities")
    if isinstance(activities, list) and activities:
        activity_df = pd.json_normalize(activities, sep=".")
        st.markdown("### Activity list")
        st.dataframe(activity_df, hide_index=True, use_container_width=True)

with vitals_tab:
    st.markdown("## Vitals & wellness")
    st.caption("Sensor availability depends on your Garmin model, settings, wear time and account features.")
    st.plotly_chart(make_line_chart(history_df, [("Resting HR (bpm)", "Resting HR (bpm)"), ("HRV (ms)", "HRV (ms)")]), use_container_width=True, config={"displayModeBar": False})
    section_payload_table(payloads, [
        "daily_stats", "hr_daily_stats", "heart_rate_samples", "hrv", "respiration", "respiration_daily_stats",
        "spo2", "all_day_stress", "stress_daily_stats", "hydration", "blood_pressure",
        "daily_weigh_ins", "body_composition", "glucose", "lifestyle_log", "health_snapshots",
        "all_day_events", "fitness_age_details", "heat_altitude_acclimation",
    ])

with all_data_tab:
    st.markdown("## Garmin data explorer")
    st.markdown("Every field returned by the endpoints requested in this session is retained below, including fields not used in the headline cards.")
    st.caption("Coverage is endpoint-dependent: this explorer does not call all Garmin Connect API methods, and your device/account may not expose every category.")
    source_names = list(payloads.keys())
    chosen_source = st.selectbox("Inspect data source", options=source_names, format_func=lambda key: f"{key} · {'available' if is_present(payloads.get(key)) else 'no data'}")
    source_value = payloads.get(chosen_source)
    if is_present(source_value):
        flat_rows = flatten_payload(chosen_source, source_value)
        st.dataframe(pd.DataFrame(flat_rows), hide_index=True, use_container_width=True, height=420)
        with st.expander("Raw response JSON", expanded=False):
            st.json(source_value, expanded=False)
        st.download_button(
            "Download this source as JSON",
            data=safe_json_download({chosen_source: source_value}),
            file_name=f"garmin_{chosen_source}_{selected_date.isoformat()}.json",
            mime="application/json",
            use_container_width=True,
        )
    else:
        st.markdown('<span class="no-data">No data returned for this source on the selected date</span>', unsafe_allow_html=True)
        if chosen_source in endpoint_errors:
            st.caption(endpoint_errors[chosen_source])
    st.download_button(
        "Download all selected-day Garmin responses",
        data=safe_json_download({"date": selected_date.isoformat(), "payloads": payloads, "endpoint_errors": endpoint_errors}),
        file_name=f"garmin_all_sources_{selected_date.isoformat()}.json",
        mime="application/json",
        use_container_width=True,
    )
    render_source_status(payloads, endpoint_errors)

with coach_tab:
    st.markdown("## Gemini performance coach")
    st.markdown("<div class='coach-panel'>Short, evidence-led feedback. Your goals come first; recommendations are tied to the readings Garmin actually returned.</div>", unsafe_allow_html=True)
    saved_short_goal = str(st.session_state.get("goal_short", ""))
    saved_long_goal = str(st.session_state.get("goal_long", ""))

    with st.form("goal_setup_form"):
        st.markdown("### Start with your goals")
        st.caption("The coach will not generate recommendations until you have defined both a near-term goal and a longer-term outcome.")
        new_short_goal = st.text_input("Near-term goal (next 2–4 weeks)", value=saved_short_goal, placeholder="e.g. Improve 5k pace without worsening recovery")
        new_long_goal = st.text_area("Longer-term goal (3–6 months)", value=saved_long_goal, placeholder="e.g. Build an aerobic base for a sub-20-minute 5k", height=90)
        goals_saved = st.form_submit_button("Save goals", type="primary", use_container_width=True)
    if goals_saved:
        st.session_state["goal_short"] = new_short_goal.strip()
        st.session_state["goal_long"] = new_long_goal.strip()
        st.rerun()

    goals = {"short": saved_short_goal.strip(), "long": saved_long_goal.strip()}
    if not goals["short"] or not goals["long"]:
        st.info("First question: what do you want to improve over the next few weeks, and what longer-term performance outcome are you working toward? Enter both above to start coaching.")
    else:
        st.markdown("### Your current goals")
        st.write(f"**Near-term:** {goals['short']}")
        st.write(f"**Longer-term:** {goals['long']}")

        context["goals"] = goals
        brief_key = f"{selected_date.isoformat()}|{goals['short']}|{goals['long']}|{model_name}"
        if st.button("Generate daily coaching check-in", type="primary", use_container_width=True):
            try:
                with st.spinner("Checking your goals against available Garmin data…"):
                    brief = call_gemini(gemini_api_key, model_name, coach_prompt(goals, context))
                st.session_state.setdefault("coach_briefs", {})[brief_key] = brief
            except Exception as exc:
                st.error(f"Gemini request failed: {type(exc).__name__}: {exc}")
        existing_brief = st.session_state.get("coach_briefs", {}).get(brief_key)
        if existing_brief:
            st.markdown("### Today’s call")
            with st.container(border=True):
                st.markdown(existing_brief)

        st.divider()
        st.markdown("### Ask about your data")
        for message in st.session_state.get("chat_messages", []):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        chat_prompt_text = st.chat_input("Ask a specific question about these numbers…")
        if chat_prompt_text:
            st.session_state.setdefault("chat_messages", []).append({"role": "user", "content": chat_prompt_text})
            try:
                prompt = coach_prompt(goals, context, question=chat_prompt_text, chat_history=st.session_state["chat_messages"][:-1])
                with st.spinner("Analysing the available readings…"):
                    answer = call_gemini(gemini_api_key, model_name, prompt)
                st.session_state["chat_messages"].append({"role": "assistant", "content": answer})
                st.rerun()
            except Exception as exc:
                st.error(f"Gemini request failed: {type(exc).__name__}: {exc}")

    st.caption("This is performance coaching, not medical diagnosis. Garmin measurements can be incomplete or device-dependent; missing values are kept missing.")


# Footer / source diagnostics
st.divider()
st.caption(
    f"{selected_date.isoformat()} · {sum(1 for v in payloads.values() if is_present(v))}/{len(payloads)} data sources returned content · "
    f"{len(history_errors)} historical endpoint notices · No synthetic headline scores are used."
)
