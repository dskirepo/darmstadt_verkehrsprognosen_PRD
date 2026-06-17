from __future__ import annotations
import argparse
import json
import math
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
try:
    from xgboost import XGBClassifier, XGBRegressor
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "xgboost is required. Install it with: pip install xgboost scikit-learn pandas joblib"
    ) from exc

# ─────────────────────────────────────────────────────────────────────────────
# global constants
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_LAGS = (1, 2, 3, 6, 12, 24, 48, 72, 168)
DEFAULT_ROLLING_WINDOWS = (3, 6, 12, 24, 72, 168)
CATEGORICAL_COLS = ["Segment", "Street", "FromNode", "ToNode", "TimeCategory", "Highway"]
STATIC_NUMERIC_COLS = [
    "StorageCapacity",
    "FlowCapacity",
    "Length_Meter",
    "Lanes",
    "RoadImportance",
    "BaseSpeed_kmh",
]
SPEED_TARGET_COL = "SpeedKmh"
STAU_TARGET_COL = "StauLevel"

GERMAN_REQUIRED_COLS = {
    "Timestamp",
    "Wochentag",
    "Tageszeit_Kategorie",
    "u",
    "v",
    "key",
    "Anzahl_Autos",
    "Durchschnittsgeschwindigkeit_kmh",
    "Stau_Level",
}
CANONICAL_REQUIRED_COLS = {"Date", "Segment", "Load"}
HOTSPOT_SCHEMA_MARKER_COLS = {
    "Segment",
    "Strassenname",
    "Highway",
    "Length_Meter",
    "Lanes",
    "RoadImportance",
    "BaseSpeed_kmh",
    "FlowCapacity",
    "EffectiveCapacity",
    "CapacityRatio",
    "HotspotPressure",
    "RoutePressure",
    "SpilloverPressure",
    "IncidentActive",
    "IncidentCapacityFactor",
}
OPTIONAL_RICH_NUMERIC_COLS = [
    "Length_Meter",
    "Lanes",
    "RoadImportance",
    "BaseSpeed_kmh",
    "EffectiveCapacity",
    "CapacityRatio",
    "HotspotPressure",
    "RoutePressure",
    "SpilloverPressure",
    "IncidentActive",
    "IncidentCapacityFactor",
]

# same output schema as simulated data
SIMULATION_OUTPUT_COLS = [
    "Timestamp",
    "Date",
    "Time",
    "Minute",
    "Wochentag",
    "Day",
    "Hour",
    "Tageszeit_Kategorie",
    "IsNonWorkingDay",
    "RushHourActive",
    "Segment",
    "u",
    "v",
    "key",
    "FromNode",
    "ToNode",
    "Strassenname",
    "Highway",
    "Length_Meter",
    "Lanes",
    "RoadImportance",
    "Anzahl_Autos",
    "Load",
    "BaseSpeed_kmh",
    "Durchschnittsgeschwindigkeit_kmh",
    "SpeedKmh",
    "FlowCapacity",
    "EffectiveCapacity",
    "CapacityRatio",
    "CongestionPercent",
    "Stau_Level",
    "Congestion",
    "HotspotPressure",
    "RoutePressure",
    "SpilloverPressure",
    "IncidentActive",
    "IncidentCapacityFactor",
    "edge_idx",
]

@dataclass
class TrainedModels:
    load_model: Pipeline
    congestion_model: Optional[Pipeline]
    speed_model: Optional[Pipeline]
    stau_model: Optional[Pipeline]
    stau_label_encoder: Optional[LabelEncoder]
    feature_cols: list[str]
    categorical_cols: list[str]
    numeric_cols: list[str]
    lags: list[int]
    rolling_windows: list[int]
    threshold_percent: float
    freq_minutes: int
    table: str
    input_schema: str


def connect(db_path: str | Path) -> sqlite3.Connection:
    return sqlite3.connect(str(db_path))


def list_tables(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    return [r[0] for r in rows]


def table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in conn.execute(f'PRAGMA table_info("{table}")')}


def detect_table_and_schema(
    conn: sqlite3.Connection,
    preferred: Optional[str] = None,
) -> tuple[str, str]:
    """
    Return (table_name, schema_name), where schema_name is one of:
    - "traffic_darmstadt_hotspot" for the new richer hotspot/network generator
    - "traffic_darmstadt" for the original generated German schema
    - "segment_history" for canonical/map-compatible history
    """
    tables = list_tables(conn)
    if not tables:
        raise ValueError("The database contains no tables.")

    def classify(cols: set[str]) -> Optional[str]:
        if GERMAN_REQUIRED_COLS.issubset(cols):
            if HOTSPOT_SCHEMA_MARKER_COLS.intersection(cols):
                return "traffic_darmstadt_hotspot"
            return "traffic_darmstadt"
        if CANONICAL_REQUIRED_COLS.issubset(cols):
            return "segment_history"
        return None

    if preferred:
        if preferred not in tables:
            raise ValueError(f"Table '{preferred}' not found. Available tables: {tables}")
        cols = table_columns(conn, preferred)
        schema = classify(cols)
        if schema:
            return preferred, schema
        raise ValueError(
            f"Table '{preferred}' does not match a supported traffic schema. "
            f"Columns found: {sorted(cols)}"
        )

    # Prefer the richer new generator if present. It contains the old German columns too, so it must be detected before the original schema
    for table in tables:
        cols = table_columns(conn, table)
        if GERMAN_REQUIRED_COLS.issubset(cols) and HOTSPOT_SCHEMA_MARKER_COLS.intersection(cols):
            return table, "traffic_darmstadt_hotspot"

    for table in tables:
        if GERMAN_REQUIRED_COLS.issubset(table_columns(conn, table)):
            return table, "traffic_darmstadt"

    for table in tables:
        if CANONICAL_REQUIRED_COLS.issubset(table_columns(conn, table)):
            return table, "segment_history"

    raise ValueError(
        "Could not find a supported traffic history table. "
        f"Available tables: {tables}. Expected either German columns "
        f"{sorted(GERMAN_REQUIRED_COLS)}, richer hotspot generator columns, "
        f"or canonical columns {sorted(CANONICAL_REQUIRED_COLS)}."
    )

def read_history(
    db_path: str | Path,
    table: Optional[str] = None,
) -> tuple[pd.DataFrame, str, str]:
    with connect(db_path) as conn:
        table, schema = detect_table_and_schema(conn, table)
        df = pd.read_sql_query(f'SELECT * FROM "{table}"', conn)
    return df, table, schema



def normalize_time_category_value(value) -> str:
    """Convert old numeric category codes to stable human-readable labels."""
    if pd.isna(value):
        return "unknown"
    text = str(value).strip().lower()
    if text.endswith(".0"):
        text = text[:-2]
    mapping = {
        "1": "nacht",
        "2": "morgens",
        "3": "mittags",
        "4": "abend",
        "5": "rush-hour-1",
        "6": "rush-hour-2",
        "night": "nacht",
        "morning": "morgens",
        "midday": "mittags",
        "noon": "mittags",
        "evening": "abend",
        "rush_hour_1": "rush-hour-1",
        "rush_hour_2": "rush-hour-2",
        "rush-hour-morning": "rush-hour-1",
        "rush-hour-evening": "rush-hour-2",
    }
    return mapping.get(text, text)


def normalize_time_category_series(values: pd.Series) -> pd.Series:
    return values.apply(normalize_time_category_value).astype(str)

def infer_time_category(timestamp: pd.Series | pd.DatetimeIndex, is_non_working_day: pd.Series | np.ndarray) -> pd.Series:
    """
    Recreate the time buckets used by the uploaded DB:
    - Working days: rush-hour-1 at 07:00-09:59, rush-hour-2 at 15:00-17:59.
    - Weekends/non-working days: those hours are treated as morgens/abend instead.
    """
    converted = pd.to_datetime(timestamp)
    if isinstance(converted, pd.Series):
        ts = converted
    else:
        ts = pd.Series(converted, index=getattr(timestamp, "index", None))

    hour = ts.dt.hour
    non_work = pd.Series(np.asarray(is_non_working_day, dtype=int), index=hour.index)

    cat = pd.Series("nacht", index=hour.index, dtype="object")
    cat[(hour >= 6) & (hour <= 9)] = "morgens"
    cat[(hour >= 10) & (hour <= 14)] = "mittags"
    cat[(hour >= 15) & (hour <= 23)] = "abend"

    working = non_work == 0
    cat[working & (hour >= 7) & (hour <= 9)] = "rush-hour-1"
    cat[working & (hour >= 15) & (hour <= 17)] = "rush-hour-2"
    return cat


def make_timestamp_from_canonical(df: pd.DataFrame) -> pd.Series:
    """Create a Timestamp column from Date+Time, with a Minute-based fallback."""
    if "Date" not in df.columns:
        raise ValueError("The database must contain either Timestamp or Date.")

    if "Time" in df.columns:
        ts = pd.to_datetime(
            df["Date"].astype(str) + " " + df["Time"].astype(str),
            errors="coerce",
        )
    else:
        ts = pd.to_datetime(df["Date"], errors="coerce")

    if ts.isna().any() and "Minute" in df.columns:
        date_only = pd.to_datetime(df["Date"], errors="coerce")
        fallback = date_only + pd.to_timedelta(pd.to_numeric(df["Minute"], errors="coerce").fillna(0), unit="m")
        ts = ts.fillna(fallback)

    if ts.isna().any():
        bad = int(ts.isna().sum())
        raise ValueError(f"Could not parse Timestamp for {bad} rows.")
    return ts


def derive_storage_capacity(
    df: pd.DataFrame,
    quantile: float = 0.98,
    margin: float = 1.15,
) -> pd.Series:
    """
    The uploaded German schema has no explicit road capacity. To still compute
    capacity-like congestion features, infer a pseudo-capacity per edge from the
    historical car-count distribution.

    This is not a physical road capacity. It is a modeling scale so that
    CongestionPercent remains meaningful for the forecast/map output.
    """
    load = pd.to_numeric(df["Load"], errors="coerce").fillna(0.0)
    tmp = pd.DataFrame({"Segment": df["Segment"].astype(str), "Load": load})
    by_segment = tmp.groupby("Segment")["Load"]
    q = by_segment.transform(lambda s: s.quantile(quantile))
    max_load = by_segment.transform("max")

    capacity = np.maximum(q * margin, max_load * 1.02)
    capacity = np.maximum(capacity, 1.0)
    return pd.Series(capacity, index=df.index).astype(float)


def normalize_german_schema(
    raw: pd.DataFrame,
    capacity_quantile: float,
    capacity_margin: float,
    schema_name: str = "traffic_darmstadt",
) -> pd.DataFrame:
    """
    Normalize both the original simple generator and the richer hotspot/network
    generator to the canonical training format.

    The new generator intentionally keeps the old German columns for backward
    compatibility, but it also adds road attributes, capacity fields, and
    pressure/incident diagnostics. This function preserves those richer fields
    instead of recreating and overwriting everything from u/v/key.
    """
    df = raw.copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    if df["Timestamp"].isna().any():
        bad = int(df["Timestamp"].isna().sum())
        raise ValueError(f"Could not parse Timestamp for {bad} rows.")

    df["u"] = df["u"].astype(str)
    df["v"] = df["v"].astype(str)
    df["key"] = df["key"].astype(str)

    df["Date"] = df["Timestamp"].dt.date.astype(str)
    df["Time"] = df["Timestamp"].dt.strftime("%H:%M")
    min_ts = df["Timestamp"].min()
    minute_offsets = ((df["Timestamp"] - min_ts).dt.total_seconds() / 60.0).round().astype(int)
    df["Minute"] = minute_offsets
    df["Day"] = (minute_offsets // 1440).astype(int)
    df["Hour"] = df["Timestamp"].dt.hour + df["Timestamp"].dt.minute / 60.0

    if "Segment" in df.columns:
        df["Segment"] = df["Segment"].fillna(
            df["u"].astype(str) + "_" + df["v"].astype(str) + "_" + df["key"].astype(str)
        ).astype(str)
    else:
        df["Segment"] = df["u"].astype(str) + "_" + df["v"].astype(str) + "_" + df["key"].astype(str)

    if "Street" in df.columns:
        df["Street"] = df["Street"].fillna("unknown").astype(str)
    elif "Strassenname" in df.columns:
        df["Street"] = df["Strassenname"].fillna("Unknown").astype(str)
    else:
        df["Street"] = "edge_" + df["Segment"]

    df["FromNode"] = df["FromNode"].astype(str) if "FromNode" in df.columns else df["u"].astype(str)
    df["ToNode"] = df["ToNode"].astype(str) if "ToNode" in df.columns else df["v"].astype(str)

    if "Load" in df.columns:
        df["Load"] = pd.to_numeric(df["Load"], errors="coerce")
        fallback_load = pd.to_numeric(df["Anzahl_Autos"], errors="coerce")
        df["Load"] = df["Load"].fillna(fallback_load)
    else:
        df["Load"] = pd.to_numeric(df["Anzahl_Autos"], errors="coerce")

    if "ReadyLoad" in df.columns:
        df["ReadyLoad"] = pd.to_numeric(df["ReadyLoad"], errors="coerce").fillna(df["Load"])
    else:
        df["ReadyLoad"] = df["Load"]

    if "InTransitLoad" in df.columns:
        df["InTransitLoad"] = pd.to_numeric(df["InTransitLoad"], errors="coerce").fillna(0.0)
    else:
        df["InTransitLoad"] = 0.0

    speed_source = "SpeedKmh" if "SpeedKmh" in df.columns else "Durchschnittsgeschwindigkeit_kmh"
    df["SpeedKmh"] = pd.to_numeric(df[speed_source], errors="coerce")
    stau_source = "StauLevel" if "StauLevel" in df.columns else "Stau_Level"
    df["StauLevel"] = pd.to_numeric(df[stau_source], errors="coerce")

    # Preserve real generator capacities when available. For the new generator,
    # FlowCapacity is the base/static capacity and EffectiveCapacity is the
    # incident-adjusted capacity at a specific timestamp. StorageCapacity is kept
    # as a stable map/model scale
    if "FlowCapacity" in df.columns:
        df["FlowCapacity"] = pd.to_numeric(df["FlowCapacity"], errors="coerce")
    elif "StorageCapacity" in df.columns:
        df["FlowCapacity"] = pd.to_numeric(df["StorageCapacity"], errors="coerce")
    else:
        df["FlowCapacity"] = np.nan

    if "StorageCapacity" in df.columns:
        df["StorageCapacity"] = pd.to_numeric(df["StorageCapacity"], errors="coerce")
        df["StorageCapacity"] = df["StorageCapacity"].fillna(df["FlowCapacity"])
    else:
        df["StorageCapacity"] = df["FlowCapacity"]

    missing_capacity = df["StorageCapacity"].isna() | (df["StorageCapacity"] <= 0)
    if missing_capacity.any():
        inferred = derive_storage_capacity(df, capacity_quantile, capacity_margin)
        df.loc[missing_capacity, "StorageCapacity"] = inferred.loc[missing_capacity]
    df["FlowCapacity"] = df["FlowCapacity"].fillna(df["StorageCapacity"])

    if "EffectiveCapacity" in df.columns:
        df["EffectiveCapacity"] = pd.to_numeric(df["EffectiveCapacity"], errors="coerce").fillna(df["FlowCapacity"])
    else:
        df["EffectiveCapacity"] = df["FlowCapacity"]

    if "CongestionPercent" in df.columns:
        df["CongestionPercent"] = pd.to_numeric(df["CongestionPercent"], errors="coerce")
    else:
        df["CongestionPercent"] = np.nan

    missing_cp = df["CongestionPercent"].isna()
    if missing_cp.any():
        denom = df.get("EffectiveCapacity", df["StorageCapacity"]).replace(0, np.nan)
        df.loc[missing_cp, "CongestionPercent"] = 100.0 * df.loc[missing_cp, "Load"] / denom.loc[missing_cp]

    # Canonical meaning inside this forecasting script: Congestion is a ratio,
    # not the binary stau flag used by the new generator. Preserve the original
    # flag separately when it exists
    if "Congestion" in raw.columns:
        df["HistoricalCongestionFlag"] = pd.to_numeric(raw["Congestion"], errors="coerce")
    df["Congestion"] = df["CongestionPercent"] / 100.0

    if "CapacityRatio" in df.columns:
        df["CapacityRatio"] = pd.to_numeric(df["CapacityRatio"], errors="coerce")
    else:
        df["CapacityRatio"] = df["Congestion"]

    df["IsNonWorkingDay"] = pd.to_numeric(
        df.get("IsNonWorkingDay", (df["Timestamp"].dt.dayofweek >= 5).astype(int)),
        errors="coerce",
    ).fillna((df["Timestamp"].dt.dayofweek >= 5).astype(int)).astype(int)

    if "Wochentag" in df.columns:
        df["Wochentag"] = pd.to_numeric(df["Wochentag"], errors="coerce").fillna(df["Timestamp"].dt.dayofweek).astype(int)

    if "Tageszeit_Kategorie" in df.columns:
        df["TimeCategory"] = normalize_time_category_series(df["Tageszeit_Kategorie"])
    else:
        df["TimeCategory"] = infer_time_category(df["Timestamp"], df["IsNonWorkingDay"]).astype(str)
    df["Tageszeit_Kategorie"] = df["TimeCategory"]
    df["RushHourActive"] = df["TimeCategory"].isin(["rush-hour-1", "rush-hour-2"]).astype(int)

    if "Highway" in df.columns:
        df["Highway"] = df["Highway"].fillna("unknown").astype(str)
    for col in OPTIONAL_RICH_NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["InputSchema"] = schema_name
    return df

def normalize_segment_history_schema(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    if "Timestamp" not in df.columns:
        df["Timestamp"] = make_timestamp_from_canonical(df)
    else:
        parsed = pd.to_datetime(df["Timestamp"], errors="coerce")
        fallback = make_timestamp_from_canonical(df) if parsed.isna().any() and "Date" in df.columns else parsed
        df["Timestamp"] = parsed.fillna(fallback)

    if "Date" not in df.columns:
        df["Date"] = df["Timestamp"].dt.date.astype(str)
    if "Time" not in df.columns:
        df["Time"] = df["Timestamp"].dt.strftime("%H:%M")

    for col in ["Load", "ReadyLoad", "InTransitLoad", "StorageCapacity", "FlowCapacity", "Congestion", "CongestionPercent"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "StorageCapacity" not in df.columns:
        df["StorageCapacity"] = derive_storage_capacity(df, 0.98, 1.15)
    if "FlowCapacity" not in df.columns:
        df["FlowCapacity"] = df["StorageCapacity"]

    for col in ["Street", "FromNode", "ToNode"]:
        if col not in df.columns:
            df[col] = "unknown"
        df[col] = df[col].fillna("unknown").astype(str)

    df["Segment"] = df["Segment"].astype(str)
    if "CongestionPercent" not in df.columns:
        df["CongestionPercent"] = 100.0 * df["Load"] / df["StorageCapacity"].replace(0, np.nan)
    if "Congestion" not in df.columns:
        df["Congestion"] = df["CongestionPercent"] / 100.0
    if "ReadyLoad" not in df.columns:
        df["ReadyLoad"] = df["Load"]
    if "InTransitLoad" not in df.columns:
        df["InTransitLoad"] = 0.0

    # Optional aliases if a canonical DB already contains these values
    if "Durchschnittsgeschwindigkeit_kmh" in df.columns and "SpeedKmh" not in df.columns:
        df["SpeedKmh"] = pd.to_numeric(df["Durchschnittsgeschwindigkeit_kmh"], errors="coerce")
    if "Stau_Level" in df.columns and "StauLevel" not in df.columns:
        df["StauLevel"] = pd.to_numeric(df["Stau_Level"], errors="coerce")

    if "InputSchema" not in df.columns:
        df["InputSchema"] = "segment_history"
    return df


def normalize_history(
    raw: pd.DataFrame,
    input_schema: str,
    capacity_quantile: float,
    capacity_margin: float,
) -> pd.DataFrame:
    if input_schema in {"traffic_darmstadt", "traffic_darmstadt_hotspot"}:
        return normalize_german_schema(raw, capacity_quantile, capacity_margin, input_schema)
    if input_schema == "segment_history":
        return normalize_segment_history_schema(raw)
    raise ValueError(f"Unsupported input schema: {input_schema}")


def infer_freq_minutes(df: pd.DataFrame) -> int:
    unique_ts = pd.Series(sorted(pd.to_datetime(df["Timestamp"].dropna().unique())))
    if len(unique_ts) < 2:
        return 60
    diffs = unique_ts.diff().dropna().dt.total_seconds() / 60.0
    median = float(diffs.median())
    return max(1, int(round(median)))


def rush_hour_factor(hour: pd.Series | np.ndarray, is_non_working_day: pd.Series | np.ndarray) -> np.ndarray:
    """Continuous rush-hour intensity feature. Zeroed on non-working days."""
    hour_arr = np.asarray(hour, dtype=float)
    non_work = np.asarray(is_non_working_day, dtype=int)
    morning = np.exp(-((hour_arr - 8.0) ** 2) / (2 * 1.2**2))
    evening = np.exp(-((hour_arr - 17.0) ** 2) / (2 * 1.5**2))
    factor = 1.0 + 1.5 * morning + 1.2 * evening
    return np.where(non_work == 1, 1.0, factor)


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    ts = df["Timestamp"]

    df["HourFloat"] = ts.dt.hour + ts.dt.minute / 60.0
    df["MinuteOfDay"] = ts.dt.hour * 60 + ts.dt.minute
    df["DayOfWeek"] = ts.dt.dayofweek
    df["Wochentag"] = df["DayOfWeek"].astype(int)
    df["DayOfMonth"] = ts.dt.day
    df["Month"] = ts.dt.month
    df["IsWeekend"] = (df["DayOfWeek"] >= 5).astype(int)

    if "IsNonWorkingDay" not in df.columns:
        df["IsNonWorkingDay"] = df["IsWeekend"]
    else:
        df["IsNonWorkingDay"] = pd.to_numeric(df["IsNonWorkingDay"], errors="coerce").fillna(df["IsWeekend"]).astype(int)

    if "TimeCategory" not in df.columns:
        if "Tageszeit_Kategorie" in df.columns:
            df["TimeCategory"] = normalize_time_category_series(df["Tageszeit_Kategorie"])
        else:
            df["TimeCategory"] = infer_time_category(ts, df["IsNonWorkingDay"]).astype(str)
    else:
        missing = df["TimeCategory"].isna()
        df["TimeCategory"] = normalize_time_category_series(df["TimeCategory"])
        if missing.any():
            df.loc[missing, "TimeCategory"] = infer_time_category(ts[missing], df.loc[missing, "IsNonWorkingDay"])

    df["Tageszeit_Kategorie"] = df["TimeCategory"]
    computed_rush = df["TimeCategory"].isin(["rush-hour-1", "rush-hour-2"]).astype(int)
    if "RushHourActive" not in df.columns:
        df["RushHourActive"] = computed_rush
    else:
        df["RushHourActive"] = pd.to_numeric(df["RushHourActive"], errors="coerce")
        df["RushHourActive"] = df["RushHourActive"].fillna(computed_rush).astype(int)
        # Numeric category codes from the generators are the source of truth when present.
        df.loc[computed_rush == 1, "RushHourActive"] = 1

    df["RushHourFactor"] = rush_hour_factor(df["HourFloat"], df["IsNonWorkingDay"])

    df["MinuteSin"] = np.sin(2 * np.pi * df["MinuteOfDay"] / 1440.0)
    df["MinuteCos"] = np.cos(2 * np.pi * df["MinuteOfDay"] / 1440.0)
    df["WeekdaySin"] = np.sin(2 * np.pi * df["DayOfWeek"] / 7.0)
    df["WeekdayCos"] = np.cos(2 * np.pi * df["DayOfWeek"] / 7.0)
    df["MonthSin"] = np.sin(2 * np.pi * df["Month"] / 12.0)
    df["MonthCos"] = np.cos(2 * np.pi * df["Month"] / 12.0)
    return df


def choose_lags(unique_times: int, requested_lags: Iterable[int]) -> list[int]:
    max_reasonable = max(1, unique_times // 2)
    lags = sorted({int(lag) for lag in requested_lags if 1 <= int(lag) <= max_reasonable})
    return lags or [1]


def choose_windows(unique_times: int, requested_windows: Iterable[int]) -> list[int]:
    max_reasonable = max(2, unique_times // 2)
    windows = sorted({int(w) for w in requested_windows if 2 <= int(w) <= max_reasonable})
    return windows or [2]


def add_lag_features(df: pd.DataFrame, lags: list[int], rolling_windows: list[int]) -> pd.DataFrame:
    df = df.sort_values(["Segment", "Timestamp"]).copy()
    grouped = df.groupby("Segment", sort=False)

    for lag in lags:
        df[f"Load_lag_{lag}"] = grouped["Load"].shift(lag)
        if "CongestionPercent" in df.columns:
            df[f"CongestionPercent_lag_{lag}"] = grouped["CongestionPercent"].shift(lag)
        if SPEED_TARGET_COL in df.columns:
            df[f"SpeedKmh_lag_{lag}"] = grouped[SPEED_TARGET_COL].shift(lag)
        if STAU_TARGET_COL in df.columns:
            df[f"StauLevel_lag_{lag}"] = grouped[STAU_TARGET_COL].shift(lag)

    shifted_load = grouped["Load"].shift(1)
    for window in rolling_windows:
        roll = shifted_load.groupby(df["Segment"], sort=False).rolling(window, min_periods=1)
        df[f"Load_roll_mean_{window}"] = roll.mean().reset_index(level=0, drop=True)
        df[f"Load_roll_max_{window}"] = roll.max().reset_index(level=0, drop=True)
        df[f"Load_roll_std_{window}"] = roll.std().reset_index(level=0, drop=True)

    if SPEED_TARGET_COL in df.columns:
        shifted_speed = grouped[SPEED_TARGET_COL].shift(1)
        for window in rolling_windows:
            roll = shifted_speed.groupby(df["Segment"], sort=False).rolling(window, min_periods=1)
            df[f"SpeedKmh_roll_mean_{window}"] = roll.mean().reset_index(level=0, drop=True)
            df[f"SpeedKmh_roll_min_{window}"] = roll.min().reset_index(level=0, drop=True)

    if "StorageCapacity" in df.columns:
        df["LoadToCapacity_lag_1"] = df.get("Load_lag_1", np.nan) / df["StorageCapacity"].replace(0, np.nan)

    return df


def prepare_history(
    raw: pd.DataFrame,
    input_schema: str,
    requested_lags: Iterable[int] = DEFAULT_LAGS,
    requested_windows: Iterable[int] = DEFAULT_ROLLING_WINDOWS,
    capacity_quantile: float = 0.98,
    capacity_margin: float = 1.15,
) -> tuple[pd.DataFrame, list[int], list[int], int]:
    df = normalize_history(raw, input_schema, capacity_quantile, capacity_margin)
    required = {"Timestamp", "Segment", "Load", "StorageCapacity"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns after normalization: {missing}")

    for col in [
        "Load",
        "ReadyLoad",
        "InTransitLoad",
        "StorageCapacity",
        "FlowCapacity",
        "Congestion",
        "CongestionPercent",
        "CapacityRatio",
        "EffectiveCapacity",
        "Length_Meter",
        "Lanes",
        "RoadImportance",
        "BaseSpeed_kmh",
        "HotspotPressure",
        "RoutePressure",
        "SpilloverPressure",
        "IncidentActive",
        "IncidentCapacityFactor",
        SPEED_TARGET_COL,
        STAU_TARGET_COL,
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values(["Timestamp", "Segment"]).reset_index(drop=True)
    df = add_calendar_features(df)
    unique_times = df["Timestamp"].nunique()
    lags = choose_lags(unique_times, requested_lags)
    windows = choose_windows(unique_times, requested_windows)
    df = add_lag_features(df, lags, windows)
    freq_minutes = infer_freq_minutes(df)
    return df, lags, windows, freq_minutes


def filter_history_by_time(
    df: pd.DataFrame,
    start: Optional[str],
    end: Optional[str],
) -> pd.DataFrame:
    if not start and not end:
        return df
    out = df.copy()
    out["Timestamp"] = pd.to_datetime(out["Timestamp"], errors="coerce")
    if start:
        out = out[out["Timestamp"] >= pd.Timestamp(start)]
    if end:
        out = out[out["Timestamp"] <= pd.Timestamp(end)]
    if out.empty:
        raise ValueError("The history filters produced an empty dataset.")
    return out


def feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    calendar_cols = [
        "HourFloat",
        "MinuteOfDay",
        "DayOfWeek",
        "Wochentag",
        "DayOfMonth",
        "Month",
        "IsWeekend",
        "IsNonWorkingDay",
        "RushHourActive",
        "RushHourFactor",
        "MinuteSin",
        "MinuteCos",
        "WeekdaySin",
        "WeekdayCos",
        "MonthSin",
        "MonthCos",
    ]
    lag_prefixes = (
        "Load_lag_",
        "CongestionPercent_lag_",
        "SpeedKmh_lag_",
        "StauLevel_lag_",
        "Load_roll_",
        "SpeedKmh_roll_",
    )
    lag_cols = [c for c in df.columns if c.startswith(lag_prefixes)]
    engineered_cols = ["LoadToCapacity_lag_1"] if "LoadToCapacity_lag_1" in df.columns else []
    numeric_cols = [c for c in STATIC_NUMERIC_COLS + calendar_cols + lag_cols + engineered_cols if c in df.columns]
    categorical_cols = [c for c in CATEGORICAL_COLS if c in df.columns]
    cols = categorical_cols + numeric_cols
    return cols, categorical_cols, numeric_cols


def build_xgb_regressor(random_state: int) -> XGBRegressor:
    return XGBRegressor(
        objective="reg:squarederror",
        n_estimators=120,
        learning_rate=0.045,
        max_depth=6,
        min_child_weight=2,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        random_state=random_state,
        n_jobs=min(4, os.cpu_count() or 4),
        tree_method="hist",
        eval_metric="rmse",
    )


def build_xgb_binary_classifier(random_state: int, y: pd.Series) -> XGBClassifier:
    pos = max(1, int(y.sum()))
    neg = max(1, int(len(y) - y.sum()))
    return XGBClassifier(
        objective="binary:logistic",
        n_estimators=100,
        learning_rate=0.045,
        max_depth=5,
        min_child_weight=2,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        scale_pos_weight=neg / pos,
        random_state=random_state,
        n_jobs=min(4, os.cpu_count() or 4),
        tree_method="hist",
        eval_metric="logloss",
    )


def build_xgb_label_classifier(random_state: int, num_class: int) -> XGBClassifier:
    params = dict(
        n_estimators=100,
        learning_rate=0.045,
        max_depth=5,
        min_child_weight=2,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        random_state=random_state,
        n_jobs=min(4, os.cpu_count() or 4),
        tree_method="hist",
    )
    if num_class <= 2:
        return XGBClassifier(objective="binary:logistic", eval_metric="logloss", **params)
    return XGBClassifier(objective="multi:softprob", num_class=num_class, eval_metric="mlogloss", **params)


def make_preprocessor(categorical_cols: list[str], numeric_cols: list[str]) -> ColumnTransformer:
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
            (
                "ordinal",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    encoded_missing_value=-1,
                ),
            ),
        ]
    )
    numeric_pipe = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])

    return ColumnTransformer(
        transformers=[
            ("cat", categorical_pipe, categorical_cols),
            ("num", numeric_pipe, numeric_cols),
        ],
        remainder="drop",
    )


def chronological_train_test_split(df: pd.DataFrame, test_size: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    times = np.array(sorted(pd.to_datetime(df["Timestamp"].unique())))
    if len(times) < 4:
        raise ValueError("Need at least 4 distinct timestamps for a chronological train/test split.")
    split_idx = max(1, min(len(times) - 1, int(math.floor(len(times) * (1 - test_size)))))
    split_time = times[split_idx]
    train = df[df["Timestamp"] < split_time].copy()
    test = df[df["Timestamp"] >= split_time].copy()
    if train.empty or test.empty:
        raise ValueError("Train/test split produced an empty dataset. Add more history or change --test-size.")
    return train, test


def maybe_sample_rows(df: pd.DataFrame, max_rows: Optional[int], random_state: int) -> pd.DataFrame:
    if not max_rows or len(df) <= max_rows:
        return df
    return df.sample(n=max_rows, random_state=random_state).sort_values(["Timestamp", "Segment"])


def compute_regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan"),
    }


def train_models(
    prepared: pd.DataFrame,
    lags: list[int],
    windows: list[int],
    threshold_percent: float,
    freq_minutes: int,
    table: str,
    input_schema: str,
    test_size: float = 0.2,
    random_state: int = 42,
    max_train_rows: Optional[int] = None,
    max_test_rows: Optional[int] = None,
) -> tuple[TrainedModels, dict[str, float], pd.DataFrame]:
    feature_cols, cat_cols, num_cols = feature_columns(prepared)

    model_df = prepared.dropna(subset=["Load"]).copy()
    if "Load_lag_1" in model_df.columns:
        model_df = model_df.dropna(subset=["Load_lag_1"])

    if model_df.empty:
        raise ValueError("No usable training rows after feature engineering. Add more history.")

    train_df, test_df = chronological_train_test_split(model_df, test_size)
    train_df_fit = maybe_sample_rows(train_df, max_train_rows, random_state)
    test_df_eval = maybe_sample_rows(test_df, max_test_rows, random_state)

    pre = make_preprocessor(cat_cols, num_cols)
    load_model = Pipeline(
        steps=[
            ("preprocess", pre),
            ("model", build_xgb_regressor(random_state)),
        ]
    )
    load_model.fit(train_df_fit[feature_cols], train_df_fit["Load"])
    test_pred = np.clip(load_model.predict(test_df_eval[feature_cols]), 0, None)
    metrics = {f"load_{k}": v for k, v in compute_regression_metrics(test_df_eval["Load"], test_pred).items()}

    # Optional speed model for the uploaded German schema.
    speed_model = None
    speed_test_pred = np.full(len(test_df_eval), np.nan)
    if SPEED_TARGET_COL in model_df.columns and model_df[SPEED_TARGET_COL].notna().sum() > 10:
        speed_train = train_df_fit.dropna(subset=[SPEED_TARGET_COL])
        speed_test = test_df_eval.dropna(subset=[SPEED_TARGET_COL])
        if not speed_train.empty and not speed_test.empty:
            speed_model = Pipeline(
                steps=[
                    ("preprocess", make_preprocessor(cat_cols, num_cols)),
                    ("model", build_xgb_regressor(random_state)),
                ]
            )
            speed_model.fit(speed_train[feature_cols], speed_train[SPEED_TARGET_COL])
            speed_pred = np.clip(speed_model.predict(speed_test[feature_cols]), 0, None)
            metrics.update(
                {f"speed_{k}": v for k, v in compute_regression_metrics(speed_test[SPEED_TARGET_COL], speed_pred).items()}
            )
            speed_test_pred = np.full(len(test_df_eval), np.nan)
            speed_positions = test_df_eval.index.get_indexer(speed_test.index)
            valid_pos = speed_positions >= 0
            speed_test_pred[speed_positions[valid_pos]] = speed_pred[valid_pos]

    # Binary congestion classifier for map-compatible probability
    if "CongestionPercent" in model_df.columns:
        y_all = (model_df["CongestionPercent"] >= threshold_percent).astype(int)
    else:
        y_all = (100.0 * model_df["Load"] / model_df["StorageCapacity"].replace(0, np.nan) >= threshold_percent).astype(int)

    congestion_model = None
    if y_all.nunique() == 2:
        y_train = y_all.loc[train_df_fit.index]
        y_test = y_all.loc[test_df_eval.index]
        congestion_model = Pipeline(
            steps=[
                ("preprocess", make_preprocessor(cat_cols, num_cols)),
                ("model", build_xgb_binary_classifier(random_state, y_train)),
            ]
        )
        congestion_model.fit(train_df_fit[feature_cols], y_train)
        p = congestion_model.predict_proba(test_df_eval[feature_cols])[:, 1]
        y_hat = (p >= 0.5).astype(int)
        metrics.update(
            {
                "congestion_accuracy": float(accuracy_score(y_test, y_hat)),
                "congestion_f1": float(f1_score(y_test, y_hat, zero_division=0)),
            }
        )
        if y_test.nunique() == 2:
            metrics["congestion_roc_auc"] = float(roc_auc_score(y_test, p))
    else:
        metrics["congestion_classifier_skipped_single_class"] = 1.0

    # Direct Stau_Level classifier for the uploaded German schema
    stau_model = None
    stau_label_encoder = None
    stau_test_pred = np.full(len(test_df_eval), np.nan)
    if STAU_TARGET_COL in model_df.columns and model_df[STAU_TARGET_COL].notna().sum() > 10:
        stau_train = train_df_fit.dropna(subset=[STAU_TARGET_COL])
        stau_test = test_df_eval.dropna(subset=[STAU_TARGET_COL])
        if not stau_train.empty and not stau_test.empty and stau_train[STAU_TARGET_COL].nunique() >= 2:
            stau_label_encoder = LabelEncoder()
            y_stau_train = stau_label_encoder.fit_transform(stau_train[STAU_TARGET_COL].astype(int))
            y_stau_test = stau_label_encoder.transform(stau_test[STAU_TARGET_COL].astype(int))

            stau_model = Pipeline(
                steps=[
                    ("preprocess", make_preprocessor(cat_cols, num_cols)),
                    ("model", build_xgb_label_classifier(random_state, len(stau_label_encoder.classes_))),
                ]
            )
            stau_model.fit(stau_train[feature_cols], y_stau_train)
            encoded_pred = stau_model.predict(stau_test[feature_cols]).astype(int)
            decoded_pred = stau_label_encoder.inverse_transform(encoded_pred)
            metrics["stau_level_accuracy"] = float(accuracy_score(y_stau_test, encoded_pred))
            metrics["stau_level_f1_macro"] = float(f1_score(y_stau_test, encoded_pred, average="macro", zero_division=0))

            stau_test_pred = np.full(len(test_df_eval), np.nan)
            stau_positions = test_df_eval.index.get_indexer(stau_test.index)
            valid_pos = stau_positions >= 0
            stau_test_pred[stau_positions[valid_pos]] = decoded_pred[valid_pos]

    trained = TrainedModels(
        load_model=load_model,
        congestion_model=congestion_model,
        speed_model=speed_model,
        stau_model=stau_model,
        stau_label_encoder=stau_label_encoder,
        feature_cols=feature_cols,
        categorical_cols=cat_cols,
        numeric_cols=num_cols,
        lags=lags,
        rolling_windows=windows,
        threshold_percent=threshold_percent,
        freq_minutes=freq_minutes,
        table=table,
        input_schema=input_schema,
    )

    keep = [
        "Timestamp",
        "Segment",
        "Street",
        "FromNode",
        "ToNode",
        "Load",
        "StorageCapacity",
        "FlowCapacity",
        "Highway",
        "Length_Meter",
        "Lanes",
        "RoadImportance",
        "BaseSpeed_kmh",
        "CongestionPercent",
        SPEED_TARGET_COL,
        STAU_TARGET_COL,
    ]
    test_predictions = test_df_eval[[c for c in keep if c in test_df_eval.columns]].copy()
    test_predictions["PredictedLoad"] = test_pred
    test_predictions["PredictedCongestionPercent"] = 100.0 * test_predictions["PredictedLoad"] / test_predictions[
        "StorageCapacity"
    ].replace(0, np.nan)
    test_predictions["PredictedSpeedKmh"] = speed_test_pred
    test_predictions["PredictedStauLevel"] = stau_test_pred

    if congestion_model is not None:
        test_predictions["CongestionProbability"] = congestion_model.predict_proba(test_df_eval[feature_cols])[:, 1]
    else:
        test_predictions["CongestionProbability"] = np.nan

    return trained, metrics, test_predictions


def static_segment_frame(history: pd.DataFrame) -> pd.DataFrame:
    static_cols = [
        "Segment",
        "Street",
        "FromNode",
        "ToNode",
        "StorageCapacity",
        "FlowCapacity",
        "u",
        "v",
        "key",
        "Strassenname",
        "Highway",
        "Length_Meter",
        "Lanes",
        "RoadImportance",
        "BaseSpeed_kmh",
        "EffectiveCapacity",
        "CapacityRatio",
        "HotspotPressure",
        "RoutePressure",
        "SpilloverPressure",
        "IncidentActive",
        "IncidentCapacityFactor",
        "edge_idx",
    ]
    available = [c for c in static_cols if c in history.columns]
    return (
        history.sort_values("Timestamp")
        .groupby("Segment", as_index=False)
        .tail(1)[available]
        .sort_values("Segment")
        .reset_index(drop=True)
    )


def make_future_base_rows(static_df: pd.DataFrame, timestamp: pd.Timestamp) -> pd.DataFrame:
    rows = static_df.copy()
    rows["Timestamp"] = timestamp
    rows["Date"] = timestamp.date().isoformat()
    rows["Time"] = timestamp.strftime("%H:%M")
    rows["Load"] = np.nan
    rows[SPEED_TARGET_COL] = np.nan
    rows[STAU_TARGET_COL] = np.nan
    rows["CongestionPercent"] = np.nan
    if "FlowCapacity" in rows.columns and "StorageCapacity" not in rows.columns:
        rows["StorageCapacity"] = rows["FlowCapacity"]
    if "EffectiveCapacity" not in rows.columns and "FlowCapacity" in rows.columns:
        rows["EffectiveCapacity"] = rows["FlowCapacity"]
    rows = add_calendar_features(rows)
    return rows


def add_recursive_features_for_step(
    rows: pd.DataFrame,
    history_plus_predictions: pd.DataFrame,
    lags: list[int],
    rolling_windows: list[int],
) -> pd.DataFrame:
    """Create lag/rolling features for one future timestamp using known/predicted history."""
    rows = rows.copy()
    hist = history_plus_predictions.sort_values(["Segment", "Timestamp"])
    grouped = {seg: g for seg, g in hist.groupby("Segment", sort=False)}

    for idx, row in rows.iterrows():
        seg = row["Segment"]
        g = grouped.get(seg)
        if g is None or g.empty:
            continue

        loads = g["Load"].to_numpy(dtype=float)
        congs = g["CongestionPercent"].to_numpy(dtype=float) if "CongestionPercent" in g.columns else None
        speeds = g[SPEED_TARGET_COL].to_numpy(dtype=float) if SPEED_TARGET_COL in g.columns else None
        staus = g[STAU_TARGET_COL].to_numpy(dtype=float) if STAU_TARGET_COL in g.columns else None

        for lag in lags:
            if len(loads) >= lag:
                rows.at[idx, f"Load_lag_{lag}"] = loads[-lag]
            if congs is not None and len(congs) >= lag:
                rows.at[idx, f"CongestionPercent_lag_{lag}"] = congs[-lag]
            if speeds is not None and len(speeds) >= lag:
                rows.at[idx, f"SpeedKmh_lag_{lag}"] = speeds[-lag]
            if staus is not None and len(staus) >= lag:
                rows.at[idx, f"StauLevel_lag_{lag}"] = staus[-lag]

        for window in rolling_windows:
            recent_loads = loads[-window:]
            if len(recent_loads) > 0:
                rows.at[idx, f"Load_roll_mean_{window}"] = float(np.nanmean(recent_loads))
                rows.at[idx, f"Load_roll_max_{window}"] = float(np.nanmax(recent_loads))
                rows.at[idx, f"Load_roll_std_{window}"] = float(np.nanstd(recent_loads, ddof=1)) if len(recent_loads) > 1 else 0.0

            if speeds is not None:
                recent_speeds = speeds[-window:]
                if len(recent_speeds) > 0:
                    rows.at[idx, f"SpeedKmh_roll_mean_{window}"] = float(np.nanmean(recent_speeds))
                    rows.at[idx, f"SpeedKmh_roll_min_{window}"] = float(np.nanmin(recent_speeds))

    if "Load_lag_1" in rows.columns and "StorageCapacity" in rows.columns:
        rows["LoadToCapacity_lag_1"] = rows["Load_lag_1"] / rows["StorageCapacity"].replace(0, np.nan)
    return rows


def date_range_minutes(start: pd.Timestamp, end: pd.Timestamp, step_minutes: int) -> list[pd.Timestamp]:
    if end < start:
        raise ValueError("Forecast end must be after forecast start.")
    freq = f"{step_minutes}min"
    return list(pd.date_range(start=start, end=end, freq=freq))


def derive_stau_from_ratio(congestion_percent: pd.Series) -> pd.Series:
    """
    Fallback if no direct Stau_Level classifier is available.
    This uses broad thresholds and should be considered less reliable than the
    trained Stau_Level classifier.
    """
    cp = pd.to_numeric(congestion_percent, errors="coerce").fillna(0.0)
    return pd.Series(np.select([cp < 60, cp < 100], [1, 2], default=3), index=cp.index).astype(int)


def forecast_recursive(
    models: TrainedModels,
    prepared_history: pd.DataFrame,
    start: Optional[str] = None,
    until: Optional[str] = None,
    horizon_minutes: int = 60*24,
    segments: Optional[list[str]] = None,
) -> pd.DataFrame:
    last_ts = pd.Timestamp(prepared_history["Timestamp"].max())
    first_internal_ts = last_ts + pd.Timedelta(minutes=models.freq_minutes)

    requested_start = pd.Timestamp(start) if start else first_internal_ts
    if requested_start < first_internal_ts:
        raise ValueError(
            f"Forecast start {requested_start} is inside the known history. "
            f"Use a time after {last_ts}."
        )

    requested_end = pd.Timestamp(until) if until else requested_start + pd.Timedelta(minutes=horizon_minutes - models.freq_minutes)
    if requested_end < requested_start:
        raise ValueError("--until must not be before --forecast-start.")

    internal_times = date_range_minutes(first_internal_ts, requested_end, models.freq_minutes)

    static_df = static_segment_frame(prepared_history)
    if segments:
        static_df = static_df[static_df["Segment"].isin(segments)].copy()
        if static_df.empty:
            raise ValueError("None of the requested --segments were found in the history table.")
        history_state = prepared_history[prepared_history["Segment"].isin(segments)].copy()
    else:
        history_state = prepared_history.copy()

    predictions: list[pd.DataFrame] = []
    for ts in internal_times:
        rows = make_future_base_rows(static_df, ts)
        rows = add_recursive_features_for_step(rows, history_state, models.lags, models.rolling_windows)

        pred_load = np.clip(models.load_model.predict(rows[models.feature_cols]), 0, None)
        rows["PredictedLoad"] = pred_load
        rows["Load"] = pred_load

        rows["PredictedCongestion"] = rows["PredictedLoad"] / rows["StorageCapacity"].replace(0, np.nan)
        rows["PredictedCongestionPercent"] = 100.0 * rows["PredictedCongestion"]
        rows["Congestion"] = rows["PredictedCongestion"]
        rows["CongestionPercent"] = rows["PredictedCongestionPercent"]

        if models.speed_model is not None:
            pred_speed = np.clip(models.speed_model.predict(rows[models.feature_cols]), 0, None)
            rows["PredictedSpeedKmh"] = pred_speed
            rows[SPEED_TARGET_COL] = pred_speed
        else:
            rows["PredictedSpeedKmh"] = np.nan

        if models.stau_model is not None and models.stau_label_encoder is not None:
            encoded = models.stau_model.predict(rows[models.feature_cols]).astype(int)
            pred_stau = models.stau_label_encoder.inverse_transform(encoded).astype(int)
            rows["PredictedStauLevel"] = pred_stau
        else:
            rows["PredictedStauLevel"] = derive_stau_from_ratio(rows["PredictedCongestionPercent"]).to_numpy()
        rows[STAU_TARGET_COL] = rows["PredictedStauLevel"]

        if models.congestion_model is not None:
            rows["CongestionProbability"] = models.congestion_model.predict_proba(rows[models.feature_cols])[:, 1]
            rows["PredictedCongested"] = (rows["CongestionProbability"] >= 0.5).astype(int)
        else:
            rows["CongestionProbability"] = (rows["PredictedStauLevel"] >= 2).astype(float)
            rows["PredictedCongested"] = (rows["PredictedStauLevel"] >= 2).astype(int)

        rows["ModelCreatedAt"] = datetime.now().isoformat(timespec="seconds")
        predictions.append(rows.copy())

        history_cols = sorted(set(history_state.columns).union(rows.columns))
        history_state = pd.concat(
            [history_state.reindex(columns=history_cols), rows.reindex(columns=history_cols)],
            ignore_index=True,
        )

    out = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    out = out[(out["Timestamp"] >= requested_start) & (out["Timestamp"] <= requested_end)].copy()

    keep_cols = [
        "Date",
        "Time",
        "Timestamp",
        "Wochentag",
        "TimeCategory",
        "Tageszeit_Kategorie",
        "Segment",
        "Street",
        "FromNode",
        "ToNode",
        "u",
        "v",
        "key",
        "PredictedLoad",
        "PredictedSpeedKmh",
        "PredictedStauLevel",
        "StorageCapacity",
        "FlowCapacity",
        "EffectiveCapacity",
        "CapacityRatio",
        "Strassenname",
        "Highway",
        "Length_Meter",
        "Lanes",
        "RoadImportance",
        "BaseSpeed_kmh",
        "HotspotPressure",
        "RoutePressure",
        "SpilloverPressure",
        "IncidentActive",
        "IncidentCapacityFactor",
        "edge_idx",
        "PredictedCongestion",
        "PredictedCongestionPercent",
        "PredictedCongested",
        "CongestionProbability",
        "IsNonWorkingDay",
        "RushHourActive",
        "RushHourFactor",
        "ModelCreatedAt",
    ]
    return out[[c for c in keep_cols if c in out.columns]].reset_index(drop=True)


def make_map_segment_history(forecasts: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the detailed forecast output into the same long-format schema used by
    the simulation/map loader. In this table, Load/Congestion are predictions.
    """
    if forecasts.empty:
        return pd.DataFrame()

    frame = forecasts.copy()
    ts = pd.to_datetime(frame["Timestamp"])
    base_ts = ts.min()
    minute_offsets = ((ts - base_ts).dt.total_seconds() / 60.0).round().astype(int)

    map_df = pd.DataFrame()
    map_df["Date"] = ts.dt.date.astype(str)
    map_df["Time"] = ts.dt.strftime("%H:%M")
    map_df["Minute"] = minute_offsets
    map_df["Day"] = (minute_offsets // 1440).astype(int)
    map_df["Hour"] = ts.dt.hour + ts.dt.minute / 60.0

    for col in ["Segment", "Street", "FromNode", "ToNode"]:
        map_df[col] = frame[col] if col in frame.columns else "unknown"

    for col in ["Strassenname", "Highway", "Length_Meter", "Lanes", "RoadImportance", "BaseSpeed_kmh"]:
        if col in frame.columns:
            map_df[col] = frame[col]

    predicted_load = pd.to_numeric(frame["PredictedLoad"], errors="coerce").fillna(0.0).clip(lower=0.0)
    storage_capacity = pd.to_numeric(frame.get("StorageCapacity", np.nan), errors="coerce")
    flow_capacity = pd.to_numeric(frame.get("FlowCapacity", np.nan), errors="coerce")

    map_df["Load"] = predicted_load
    map_df["ReadyLoad"] = predicted_load
    map_df["InTransitLoad"] = 0.0
    map_df["StorageCapacity"] = storage_capacity
    map_df["FlowCapacity"] = flow_capacity

    if "PredictedCongestion" in frame.columns:
        map_df["Congestion"] = pd.to_numeric(frame["PredictedCongestion"], errors="coerce")
    else:
        map_df["Congestion"] = predicted_load / storage_capacity.replace(0, np.nan)

    if "PredictedCongestionPercent" in frame.columns:
        map_df["CongestionPercent"] = pd.to_numeric(frame["PredictedCongestionPercent"], errors="coerce")
    else:
        map_df["CongestionPercent"] = 100.0 * map_df["Congestion"]

    map_df["IsNonWorkingDay"] = pd.to_numeric(frame.get("IsNonWorkingDay", 0), errors="coerce").fillna(0).astype(int)
    map_df["RushHourActive"] = pd.to_numeric(frame.get("RushHourActive", 1), errors="coerce").fillna(1).astype(int)

    map_df["IsForecast"] = 1
    if "PredictedCongested" in frame.columns:
        map_df["PredictedCongested"] = pd.to_numeric(frame["PredictedCongested"], errors="coerce")
    if "CongestionProbability" in frame.columns:
        map_df["CongestionProbability"] = pd.to_numeric(frame["CongestionProbability"], errors="coerce")
    if "PredictedSpeedKmh" in frame.columns:
        map_df["PredictedSpeedKmh"] = pd.to_numeric(frame["PredictedSpeedKmh"], errors="coerce")
    if "PredictedStauLevel" in frame.columns:
        map_df["PredictedStauLevel"] = pd.to_numeric(frame["PredictedStauLevel"], errors="coerce")
    if "ModelCreatedAt" in frame.columns:
        map_df["ModelCreatedAt"] = frame["ModelCreatedAt"]

    ordered = [
        "Date", "Time", "Minute", "Day", "Hour", "Segment", "Street", "FromNode", "ToNode",
        "Load", "ReadyLoad", "InTransitLoad", "StorageCapacity", "FlowCapacity",
        "Strassenname", "Highway", "Length_Meter", "Lanes", "RoadImportance", "BaseSpeed_kmh",
        "Congestion", "CongestionPercent", "IsNonWorkingDay", "RushHourActive",
        "IsForecast", "PredictedCongested", "CongestionProbability", "PredictedSpeedKmh",
        "PredictedStauLevel", "ModelCreatedAt",
    ]
    return map_df[[c for c in ordered if c in map_df.columns]]


def simulation_time_category_codes(timestamp: pd.Series | pd.DatetimeIndex) -> pd.Series:
    """Return the numeric Tageszeit_Kategorie codes used by the simulator."""
    converted = pd.to_datetime(timestamp)
    if isinstance(converted, pd.Series):
        ts = converted
    else:
        ts = pd.Series(converted, index=getattr(timestamp, "index", None))

    hour = ts.dt.hour
    weekend = ts.dt.dayofweek >= 5
    code = pd.Series(4, index=ts.index, dtype="int64")
    code[(hour >= 0) & (hour < 6)] = 1
    code[(hour >= 6) & (hour < 10)] = 2
    code[(hour >= 10) & (hour < 15)] = 3
    code[(hour >= 15) & (hour < 24)] = 4

    working = ~weekend
    code[working & (hour >= 7) & (hour < 10)] = 5
    code[working & (hour >= 15) & (hour < 18)] = 6
    return code.astype(int)


def make_input_schema_forecast(forecasts: pd.DataFrame, input_schema: str) -> pd.DataFrame:
    """
    Create a forecast table that resembles the input DB schema. For the uploaded
    German schema this means the output can be consumed by code expecting columns:
    Timestamp, Wochentag, Tageszeit_Kategorie, u, v, key,
    Anzahl_Autos, Durchschnittsgeschwindigkeit_kmh, Stau_Level
    """
    if forecasts.empty:
        return pd.DataFrame(columns=SIMULATION_OUTPUT_COLS)

    frame = forecasts.copy()
    ts = pd.to_datetime(frame["Timestamp"])

    def numeric_series(col: str, default=np.nan) -> pd.Series:
        if col in frame.columns:
            return pd.to_numeric(frame[col], errors="coerce")
        return pd.Series(default, index=frame.index, dtype="float64")

    def object_series(col: str, default="unknown") -> pd.Series:
        if col in frame.columns:
            return frame[col]
        return pd.Series(default, index=frame.index, dtype="object")

    out = pd.DataFrame(index=frame.index)

    time_category = simulation_time_category_codes(ts)
    is_non_working_day = (ts.dt.dayofweek >= 5).astype(int)
    predicted_load = numeric_series("PredictedLoad", 0.0).fillna(0.0).clip(lower=0.0)
    load_int = predicted_load.round().astype(int)

    flow_capacity = numeric_series("FlowCapacity")
    storage_capacity = numeric_series("StorageCapacity")
    flow_capacity = flow_capacity.fillna(storage_capacity)
    flow_capacity = flow_capacity.fillna(1.0).replace(0, 1.0)

    effective_capacity = numeric_series("EffectiveCapacity").fillna(flow_capacity)
    effective_capacity = effective_capacity.mask(effective_capacity <= 0, flow_capacity)
    capacity_ratio = predicted_load / effective_capacity.replace(0, np.nan)

    pred_speed = numeric_series("PredictedSpeedKmh")
    base_speed = numeric_series("BaseSpeed_kmh", 30.0).fillna(30.0)
    speed = pred_speed.fillna(base_speed).clip(lower=0.0).round(1)

    pred_stau = numeric_series("PredictedStauLevel")
    if pred_stau.isna().any():
        pred_stau = pred_stau.fillna(derive_stau_from_ratio(100.0 * capacity_ratio).astype(float))
    stau_level = pred_stau.round().fillna(1).clip(lower=1, upper=4).astype(int)

    out["Timestamp"] = pd.to_datetime(ts)
    out["Date"] = ts.dt.date.astype(str)
    out["Time"] = ts.dt.strftime("%H:%M:%S")
    out["Minute"] = ((ts - ts.min()).dt.total_seconds() / 60.0).round().astype(int)
    out["Wochentag"] = ts.dt.dayofweek.astype(int)
    # In simuls_data_hotspot_network.py, Day is also the day-of-week value.
    out["Day"] = out["Wochentag"]
    out["Hour"] = ts.dt.hour.astype(int)
    out["Tageszeit_Kategorie"] = time_category.to_numpy()
    out["IsNonWorkingDay"] = is_non_working_day
    out["RushHourActive"] = time_category.isin([5, 6]).astype(int).to_numpy()

    out["Segment"] = object_series("Segment", "unknown").astype(str)
    for col in ["u", "v", "key"]:
        out[col] = numeric_series(col, -1).fillna(-1).astype(int)

    if "FromNode" in frame.columns:
        out["FromNode"] = numeric_series("FromNode").fillna(out["u"]).astype(int)
    else:
        out["FromNode"] = out["u"]
    if "ToNode" in frame.columns:
        out["ToNode"] = numeric_series("ToNode").fillna(out["v"]).astype(int)
    else:
        out["ToNode"] = out["v"]

    if "Strassenname" in frame.columns:
        out["Strassenname"] = object_series("Strassenname", "Unknown").fillna("Unknown").astype(str)
    elif "Street" in frame.columns:
        out["Strassenname"] = object_series("Street", "Unknown").fillna("Unknown").astype(str)
    else:
        out["Strassenname"] = "Unknown"

    out["Highway"] = object_series("Highway", "unknown").fillna("unknown").astype(str)
    out["Length_Meter"] = numeric_series("Length_Meter").round(2)
    out["Lanes"] = numeric_series("Lanes", 1.0).fillna(1.0)
    out["RoadImportance"] = numeric_series("RoadImportance", 1.0).fillna(1.0)
    out["Anzahl_Autos"] = load_int
    out["Load"] = load_int
    out["BaseSpeed_kmh"] = base_speed.round(1)
    out["Durchschnittsgeschwindigkeit_kmh"] = speed
    out["SpeedKmh"] = speed
    out["FlowCapacity"] = flow_capacity.round(1)
    out["EffectiveCapacity"] = effective_capacity.round(1)
    out["CapacityRatio"] = capacity_ratio.round(4)
    out["CongestionPercent"] = np.round(np.clip(capacity_ratio, 0.0, 2.0) * 100.0, 1)
    out["Stau_Level"] = stau_level
    # Match the simulator semantics: Congestion is a binary flag, not a ratio.
    out["Congestion"] = (stau_level >= 3).astype(int)

    out["HotspotPressure"] = numeric_series("HotspotPressure", 0.0).fillna(0.0).round(2)
    out["RoutePressure"] = numeric_series("RoutePressure", 0.0).fillna(0.0).round(2)
    out["SpilloverPressure"] = numeric_series("SpilloverPressure", 0.0).fillna(0.0).round(2)
    # Future incidents are not forecast by this model; keep neutral values.
    out["IncidentActive"] = 0
    out["IncidentCapacityFactor"] = 1.0
    out["edge_idx"] = numeric_series("edge_idx", -1).fillna(-1).astype(int)

    return out[SIMULATION_OUTPUT_COLS]


def write_outputs(
    output_db: str | Path,
    forecasts: pd.DataFrame,
    input_schema: str,
    output_table: str = "traffic_darmstadt",
    write_chunk_size: int = 50_000,
) -> None:
    output_db = Path(output_db)
    output_db.parent.mkdir(parents=True, exist_ok=True)
    output_table = output_table.strip() or "traffic_darmstadt"

    simulation_schema_df = make_input_schema_forecast(forecasts, input_schema)

    with connect(output_db) as conn:
        if simulation_schema_df.empty:
            simulation_schema_df.to_sql(output_table, conn, if_exists="replace", index=False)
        else:
            first = True
            for start in range(0, len(simulation_schema_df), write_chunk_size):
                end = start + write_chunk_size
                chunk = simulation_schema_df.iloc[start:end]
                chunk.to_sql(
                    output_table,
                    conn,
                    if_exists="replace" if first else "append",
                    index=False,
                )
                first = False

        # Same index pattern as simul_data.py
        conn.execute(f'CREATE INDEX IF NOT EXISTS idx_{output_table}_time ON "{output_table}"(Timestamp)')
        conn.execute(f'CREATE INDEX IF NOT EXISTS idx_{output_table}_segment ON "{output_table}"(Segment)')
        conn.execute(
            f'CREATE INDEX IF NOT EXISTS idx_{output_table}_datetime_segment '
            f'ON "{output_table}"(Date, Time, Segment)'
        )


def save_models(model_dir: str | Path, models: TrainedModels, metrics: dict[str, float]) -> None:
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(models.load_model, model_dir / "load_model.joblib")
    if models.congestion_model is not None:
        joblib.dump(models.congestion_model, model_dir / "congestion_model.joblib")
    if models.speed_model is not None:
        joblib.dump(models.speed_model, model_dir / "speed_model.joblib")
    if models.stau_model is not None:
        joblib.dump(models.stau_model, model_dir / "stau_level_model.joblib")
    if models.stau_label_encoder is not None:
        joblib.dump(models.stau_label_encoder, model_dir / "stau_label_encoder.joblib")

    metadata = {
        "feature_cols": models.feature_cols,
        "categorical_cols": models.categorical_cols,
        "numeric_cols": models.numeric_cols,
        "lags": models.lags,
        "rolling_windows": models.rolling_windows,
        "threshold_percent": models.threshold_percent,
        "freq_minutes": models.freq_minutes,
        "table": models.table,
        "input_schema": models.input_schema,
        "metrics": metrics,
    }
    (model_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def parse_int_list(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_str_list(text: Optional[str]) -> Optional[list[str]]:
    if not text:
        return None
    return [x.strip() for x in text.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an XGBoost traffic forecast model from a SQLite traffic DB.")
    parser.add_argument("--db", default="traffic_data_darmstadt_mitte.db", help="Input SQLite database path.")
    parser.add_argument("--table", default=None, help="History table name. Defaults to auto-detect.")
    parser.add_argument("--output-db", default="traffic_forecasts.db", help="SQLite DB to write the forecast table into.")
    parser.add_argument("--output-table", default="traffic_darmstadt", help="Single output table name. Defaults to the simulator table name.")
    parser.add_argument("--model-dir", default="traffic_xgb_models", help="Directory for trained model files, used only with --save-models.")
    parser.add_argument("--save-models", action="store_true", help="Optionally save trained model files. Disabled by default so only the DB is created.")

    parser.add_argument("--forecast-start", default=None, help='First desired forecast timestamp, e.g. "2026-05-01 01:00".')
    parser.add_argument("--until", default=None, help='Last desired forecast timestamp, e.g. "2026-05-02 00:00".')
    parser.add_argument("--horizon-minutes", type=int, default =60 *24, help="Forecast horizon in minutes if --until is omitted.")

    parser.add_argument("--history-start", default=None, help="Optional lower bound for training history.")
    parser.add_argument("--history-end", default=None, help="Optional upper bound for training history.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Chronological holdout fraction.")
    parser.add_argument("--max-train-rows", type=int, default=None, help="Optional row cap for faster training.")
    parser.add_argument("--max-test-rows", type=int, default=None, help="Optional row cap for faster metric calculation.")

    parser.add_argument("--congestion-threshold-percent", type=float, default=100.0, help="Capacity threshold for binary congestion label.")
    parser.add_argument("--capacity-quantile", type=float, default=0.98, help="Quantile used to infer pseudo-capacity for DBs without capacity.")
    parser.add_argument("--capacity-margin", type=float, default=1.15, help="Multiplier used after the capacity quantile.")

    parser.add_argument("--lags", default=",".join(map(str, DEFAULT_LAGS)), help="Comma-separated lag steps. With hourly data, lag 24 = yesterday.")
    parser.add_argument("--rolling-windows", default=",".join(map(str, DEFAULT_ROLLING_WINDOWS)), help="Comma-separated rolling windows.")
    parser.add_argument("--segments", default=None, help="Optional comma-separated Segment IDs to forecast/output, e.g. '447856_12635609_0'.")

    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw, table, input_schema = read_history(args.db, args.table)
    raw = filter_history_by_time(raw, args.history_start, args.history_end) if "Timestamp" in raw.columns else raw

    prepared, lags, windows, freq_minutes = prepare_history(
        raw=raw,
        input_schema=input_schema,
        requested_lags=parse_int_list(args.lags),
        requested_windows=parse_int_list(args.rolling_windows),
        capacity_quantile=args.capacity_quantile,
        capacity_margin=args.capacity_margin,
    )

    print(f"Loaded {len(raw):,} rows from table '{table}' using schema '{input_schema}'.")
    print(f"Found {prepared['Segment'].nunique():,} segments and {prepared['Timestamp'].nunique():,} timestamps.")
    print(f"Inferred step size: {freq_minutes} minute(s). Using lags={lags}, rolling_windows={windows}.")
    if input_schema in {"traffic_darmstadt", "traffic_darmstadt_hotspot"}:
        print("Mapped Anzahl_Autos -> Load, Durchschnittsgeschwindigkeit_kmh -> SpeedKmh, Stau_Level -> StauLevel.")
    if input_schema == "traffic_darmstadt_hotspot":
        print("Detected richer hotspot/network generator fields and preserved road attributes for training/output.")

    models, metrics, test_predictions = train_models(
        prepared=prepared,
        lags=lags,
        windows=windows,
        threshold_percent=args.congestion_threshold_percent,
        freq_minutes=freq_minutes,
        table=table,
        input_schema=input_schema,
        test_size=args.test_size,
        random_state=args.random_state,
        max_train_rows=args.max_train_rows,
        max_test_rows=args.max_test_rows,
    )

    print("Holdout metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    forecasts = forecast_recursive(
        models=models,
        prepared_history=prepared,
        start=args.forecast_start,
        until=args.until,
        horizon_minutes=args.horizon_minutes,
        segments=parse_str_list(args.segments),
    )

    write_outputs(
        args.output_db,
        forecasts,
        input_schema=input_schema,
        output_table=args.output_table,
    )

    if args.save_models:
        save_models(args.model_dir, models, metrics)

    print(f"Wrote {len(forecasts):,} forecast rows to {args.output_db} table '{args.output_table}'.")
    print("The output database contains only the simulator-schema forecast table.")
    if args.save_models:
        print(f"Saved models and metadata in {args.model_dir}.")


if __name__ == "__main__":
    main()
