from __future__ import annotations

import argparse
import math
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

try:
    from xgboost import XGBRegressor
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "xgboost is required. Install it with: pip install xgboost scikit-learn pandas"
    ) from exc



DEFAULT_LAGS = (1, 2, 3, 6, 12, 24, 48, 72, 168)
DEFAULT_ROLLING_WINDOWS = (3, 6, 12, 24, 72, 168)

TABLE_NAME = "traffic_darmstadt"
INPUT_DB = "traffic_data_darmstadt_mitte.db"
OUTPUT_DB = "traffic_forecasts_new.db"
# The output table is written with the same column order and date/time types as the input table.

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

CATEGORICAL_COLS = ["Segment", "Strassenname", "Highway"]
STATIC_NUMERIC_COLS = [
    "u",
    "v",
    "key",
    "edge_idx",
    "FromNode",
    "ToNode",
    "Length_Meter",
    "Lanes",
    "RoadImportance",
    "BaseSpeed_kmh",
    "FlowCapacity",
]
CALENDAR_NUMERIC_COLS = [
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
LAG_PREFIXES = (
    "Load_lag_",
    "SpeedKmh_lag_",
    "Stau_Level_lag_",
    "CongestionPercent_lag_",
    "Load_roll_",
    "SpeedKmh_roll_",
)


@dataclass
class ForecastModels:
    load_model: Pipeline
    speed_model: Pipeline
    feature_cols: list[str]
    lags: list[int]
    rolling_windows: list[int]
    freq_minutes: int
    history_start_ts: pd.Timestamp

#easily-enough understood methods to make the main code less large.
def connect(db_path: str | Path) -> sqlite3.Connection:
    return sqlite3.connect(str(db_path))


def parse_int_list(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_str_list(text: Optional[str]) -> Optional[list[str]]:
    if not text:
        return None
    return [x.strip() for x in text.split(",") if x.strip()]


def read_history(db_path: str | Path, table: str = TABLE_NAME) -> pd.DataFrame:
    with connect(db_path) as conn:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if table not in tables:
            raise ValueError(f"Table '{table}' not found. Available tables: {sorted(tables)}")
        df = pd.read_sql_query(f'SELECT * FROM "{table}"', conn)
    return df


def validate_simulation_schema(df: pd.DataFrame) -> None:
    missing = [c for c in SIMULATION_OUTPUT_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            "The input DB does not match the schema produced by simul_data.py. "
            f"Missing columns: {missing}"
        )


def normalize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Use the same practical dtypes as the simulator before training/output."""
    out = df.copy()
    out["Timestamp"] = pd.to_datetime(out["Timestamp"], errors="coerce")
    if out["Timestamp"].isna().any():
        raise ValueError(f"Could not parse Timestamp for {int(out['Timestamp'].isna().sum())} rows.")

    text_cols = ["Date", "Time", "Segment", "Strassenname", "Highway"]
    int_cols = [
        "Minute",
        "Wochentag",
        "Day",
        "Hour",
        "Tageszeit_Kategorie",
        "IsNonWorkingDay",
        "RushHourActive",
        "u",
        "v",
        "key",
        "FromNode",
        "ToNode",
        "Anzahl_Autos",
        "Load",
        "Stau_Level",
        "Congestion",
        "IncidentActive",
        "edge_idx",
    ]
    float_cols = [
        "Length_Meter",
        "Lanes",
        "RoadImportance",
        "BaseSpeed_kmh",
        "Durchschnittsgeschwindigkeit_kmh",
        "SpeedKmh",
        "FlowCapacity",
        "EffectiveCapacity",
        "CapacityRatio",
        "CongestionPercent",
        "HotspotPressure",
        "RoutePressure",
        "SpilloverPressure",
        "IncidentCapacityFactor",
    ]

    for col in text_cols:
        out[col] = out[col].fillna("Unknown").astype(str)
    for col in int_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).round().astype("int64")
    for col in float_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")

    # The simulator stores FromNode/ToNode as aliases of u/v.
    out["FromNode"] = out["u"].astype("int64")
    out["ToNode"] = out["v"].astype("int64")
    return out[SIMULATION_OUTPUT_COLS]


def get_time_category(hour: int, day_of_week: int) -> int:
    """
    Same category logic as simul_data.py:
    1 night, 2 morning, 3 midday, 4 evening, 5 morning rush, 6 evening rush.
    """
    if day_of_week >= 5:
        if 0 <= hour < 6:
            return 1
        if 6 <= hour < 10:
            return 2
        if 10 <= hour < 15:
            return 3
        return 4

    if 0 <= hour < 6:
        return 1
    if 6 <= hour < 7:
        return 2
    if 7 <= hour < 10:
        return 5
    if 10 <= hour < 15:
        return 3
    if 15 <= hour < 18:
        return 6
    return 4


def rush_hour_factor(hour: pd.Series | np.ndarray, is_non_working_day: pd.Series | np.ndarray) -> np.ndarray:
    """Smooth rush-hour intensity used only as an ML feature. We can do this because we know
    the nature of our data, namely traffic. This is not just writing out part of the solution in 
    the forecast, as it can be reasonably expected that a traffic forecast programm would know about 
    the concept of rush-hour, making this reasonable."""
    hour_arr = np.asarray(hour, dtype=float)
    non_work = np.asarray(is_non_working_day, dtype=int)
    morning = np.exp(-((hour_arr - 8.0) ** 2) / (2 * 1.2**2))
    evening = np.exp(-((hour_arr - 17.0) ** 2) / (2 * 1.5**2))
    factor = 1.0 + 1.5 * morning + 1.2 * evening
    return np.where(non_work == 1, 1.0, factor)


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    ts = pd.to_datetime(out["Timestamp"], errors="coerce")
    out["HourFloat"] = ts.dt.hour + ts.dt.minute / 60.0
    out["MinuteOfDay"] = ts.dt.hour * 60 + ts.dt.minute
    out["DayOfWeek"] = ts.dt.dayofweek
    out["Wochentag"] = out["DayOfWeek"].astype("int64")
    out["DayOfMonth"] = ts.dt.day
    out["Month"] = ts.dt.month
    out["IsWeekend"] = (out["DayOfWeek"] >= 5).astype("int64")
    out["IsNonWorkingDay"] = out["IsWeekend"].astype("int64")
    out["Tageszeit_Kategorie"] = [get_time_category(int(h), int(d)) for h, d in zip(ts.dt.hour, out["DayOfWeek"])]
    out["RushHourActive"] = out["Tageszeit_Kategorie"].isin([5, 6]).astype("int64")
    out["RushHourFactor"] = rush_hour_factor(out["HourFloat"], out["IsNonWorkingDay"])

    out["MinuteSin"] = np.sin(2 * np.pi * out["MinuteOfDay"] / 1440.0)
    out["MinuteCos"] = np.cos(2 * np.pi * out["MinuteOfDay"] / 1440.0)
    out["WeekdaySin"] = np.sin(2 * np.pi * out["DayOfWeek"] / 7.0)
    out["WeekdayCos"] = np.cos(2 * np.pi * out["DayOfWeek"] / 7.0)
    out["MonthSin"] = np.sin(2 * np.pi * out["Month"] / 12.0)
    out["MonthCos"] = np.cos(2 * np.pi * out["Month"] / 12.0)
    return out


def infer_freq_minutes(df: pd.DataFrame) -> int:
    """Infers the step size, as in, how far away predicted datapoints are from each other in time. This is done based on the step size of the historical data."""
    times = pd.Series(sorted(pd.to_datetime(df["Timestamp"].dropna().unique())))
    if len(times) < 2:
        return 60
    diffs = times.diff().dropna().dt.total_seconds() / 60.0
    return max(1, int(round(float(diffs.median()))))


def choose_lags(unique_times: int, requested_lags: Iterable[int]) -> list[int]:
    max_reasonable = max(1, unique_times // 2)
    lags = sorted({int(lag) for lag in requested_lags if 1 <= int(lag) <= max_reasonable})
    return lags or [1]


def choose_windows(unique_times: int, requested_windows: Iterable[int]) -> list[int]:
    max_reasonable = max(2, unique_times // 2)
    windows = sorted({int(w) for w in requested_windows if 2 <= int(w) <= max_reasonable})
    return windows or [2]


def add_lag_features(df: pd.DataFrame, lags: list[int], rolling_windows: list[int]) -> pd.DataFrame:
    out = df.sort_values(["Segment", "Timestamp"]).copy()
    grouped = out.groupby("Segment", sort=False)

    for lag in lags:
        out[f"Load_lag_{lag}"] = grouped["Load"].shift(lag)
        out[f"SpeedKmh_lag_{lag}"] = grouped["SpeedKmh"].shift(lag)
        out[f"Stau_Level_lag_{lag}"] = grouped["Stau_Level"].shift(lag)
        out[f"CongestionPercent_lag_{lag}"] = grouped["CongestionPercent"].shift(lag)

    shifted_load = grouped["Load"].shift(1)
    for window in rolling_windows:
        roll = shifted_load.groupby(out["Segment"], sort=False).rolling(window, min_periods=1)
        out[f"Load_roll_mean_{window}"] = roll.mean().reset_index(level=0, drop=True)
        out[f"Load_roll_max_{window}"] = roll.max().reset_index(level=0, drop=True)
        out[f"Load_roll_std_{window}"] = roll.std().reset_index(level=0, drop=True)

    shifted_speed = grouped["SpeedKmh"].shift(1)
    for window in rolling_windows:
        roll = shifted_speed.groupby(out["Segment"], sort=False).rolling(window, min_periods=1)
        out[f"SpeedKmh_roll_mean_{window}"] = roll.mean().reset_index(level=0, drop=True)
        out[f"SpeedKmh_roll_min_{window}"] = roll.min().reset_index(level=0, drop=True)

    out["LoadToCapacity_lag_1"] = out.get("Load_lag_1", np.nan) / out["FlowCapacity"].replace(0, np.nan)
    return out


def prepare_history(
    raw: pd.DataFrame,
    requested_lags: Iterable[int],
    requested_windows: Iterable[int],
) -> tuple[pd.DataFrame, list[int], list[int], int]:
    """Validates the schema of data, normalizes it, sorts by Timestamp and Segment,
    Creates Calendar features, infears the step-size, or freq_minutes, of the given data,
    usable lags and similar."""
    validate_simulation_schema(raw)
    history = normalize_dtypes(raw)
    history = history.sort_values(["Timestamp", "Segment"]).reset_index(drop=True)
    history = add_calendar_features(history)

    unique_times = history["Timestamp"].nunique()
    lags = choose_lags(unique_times, requested_lags)
    windows = choose_windows(unique_times, requested_windows)
    freq_minutes = infer_freq_minutes(history)
    history = add_lag_features(history, lags, windows)
    return history, lags, windows, freq_minutes


def filter_history_by_time(df: pd.DataFrame, start: Optional[str], end: Optional[str]) -> pd.DataFrame:
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


def feature_columns(df: pd.DataFrame) -> list[str]:
    lag_cols = [c for c in df.columns if c.startswith(LAG_PREFIXES)]
    engineered_cols = ["LoadToCapacity_lag_1"] if "LoadToCapacity_lag_1" in df.columns else []
    cols = [c for c in CATEGORICAL_COLS + STATIC_NUMERIC_COLS + CALENDAR_NUMERIC_COLS + lag_cols + engineered_cols if c in df.columns]
    return cols


def make_preprocessor(feature_cols: list[str]) -> ColumnTransformer:
    categorical_cols = [c for c in CATEGORICAL_COLS if c in feature_cols]
    numeric_cols = [c for c in feature_cols if c not in categorical_cols]

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


def chronological_train_test_split(df: pd.DataFrame, test_size: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Splits data into training and test data based on the time. Earlier datapoints become training data, later ones testing data."""
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


def regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan"),
    }


def train_models(
    prepared: pd.DataFrame,
    lags: list[int],
    windows: list[int],
    freq_minutes: int,
    test_size: float,
    random_state: int,
    max_train_rows: Optional[int],
    max_test_rows: Optional[int],
) -> tuple[ForecastModels, dict[str, float]]:
    feature_cols = feature_columns(prepared)
    #the following drops unusable rows, including the very first datapoint because it has no lag (though it indirectly has an impact via the lag of the second datapoint)
    model_df = prepared.dropna(subset=["Load", "SpeedKmh", "Load_lag_1"]).copy()
    if model_df.empty:
        raise ValueError("No usable training rows after lag feature creation.")

    train_df, test_df = chronological_train_test_split(model_df, test_size)
    #if max rows are set, can reduce total data used. 
    train_fit = maybe_sample_rows(train_df, max_train_rows, random_state)
    test_eval = maybe_sample_rows(test_df, max_test_rows, random_state)
    #creates two models, one training on "Load", the other on "SpeedKmh"
    load_model = Pipeline(
        steps=[
            ("preprocess", make_preprocessor(feature_cols)),
            ("model", build_xgb_regressor(random_state)),
        ]
    )
    load_model.fit(train_fit[feature_cols], train_fit["Load"])
    load_pred = np.clip(load_model.predict(test_eval[feature_cols]), 0, None)

    speed_model = Pipeline(
        steps=[
            ("preprocess", make_preprocessor(feature_cols)),
            ("model", build_xgb_regressor(random_state)),
        ]
    )
    speed_model.fit(train_fit[feature_cols], train_fit["SpeedKmh"])
    speed_pred = np.clip(speed_model.predict(test_eval[feature_cols]), 0, None)

    metrics = {f"load_{k}": v for k, v in regression_metrics(test_eval["Load"], load_pred).items()}
    metrics.update({f"speed_{k}": v for k, v in regression_metrics(test_eval["SpeedKmh"], speed_pred).items()})

    models = ForecastModels(
        load_model=load_model,
        speed_model=speed_model,
        feature_cols=feature_cols,
        lags=lags,
        rolling_windows=windows,
        freq_minutes=freq_minutes,
        history_start_ts=pd.Timestamp(prepared["Timestamp"].min()),
    )
    return models, metrics


def static_segment_frame(history: pd.DataFrame) -> pd.DataFrame:
    static_cols = [
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
        "BaseSpeed_kmh",
        "FlowCapacity",
        "edge_idx",
    ]
    return (
        history.sort_values("Timestamp")
        .groupby("Segment", as_index=False)
        .tail(1)[static_cols]
        .sort_values("Segment")
        .reset_index(drop=True)
    )


def make_future_base_rows(static_df: pd.DataFrame, timestamp: pd.Timestamp, history_start_ts: pd.Timestamp) -> pd.DataFrame:
    rows = static_df.copy()
    ts = pd.Timestamp(timestamp)
    day_of_week = int(ts.dayofweek)
    time_category = get_time_category(int(ts.hour), day_of_week)

    rows["Timestamp"] = ts
    rows["Date"] = ts.date().isoformat()
    rows["Time"] = ts.time().strftime("%H:%M:%S")
    rows["Minute"] = int((ts - history_start_ts).total_seconds() // 60)
    rows["Wochentag"] = day_of_week
    rows["Day"] = day_of_week
    rows["Hour"] = int(ts.hour)
    rows["Tageszeit_Kategorie"] = int(time_category)
    rows["IsNonWorkingDay"] = int(day_of_week >= 5)
    rows["RushHourActive"] = int(time_category in {5, 6})

    # Neutral future incident assumption, as incidents are not forecast by this model.
    rows["IncidentActive"] = 0
    rows["IncidentCapacityFactor"] = 1.0
    rows["EffectiveCapacity"] = rows["FlowCapacity"].astype(float)
    rows["HotspotPressure"] = 0.0
    rows["RoutePressure"] = 0.0
    rows["SpilloverPressure"] = 0.0

    # Placeholder target columns; recursive lag creation will use history_state instead.
    rows["Load"] = np.nan
    rows["Anzahl_Autos"] = np.nan
    rows["SpeedKmh"] = np.nan
    rows["Durchschnittsgeschwindigkeit_kmh"] = np.nan
    rows["Stau_Level"] = np.nan
    rows["Congestion"] = np.nan
    rows["CapacityRatio"] = np.nan
    rows["CongestionPercent"] = np.nan

    rows = add_calendar_features(rows)
    return rows


def add_recursive_features_for_step(
    rows: pd.DataFrame,
    history_state: pd.DataFrame,
    lags: list[int],
    rolling_windows: list[int],
) -> pd.DataFrame:
    rows = rows.copy()
    hist = history_state.sort_values(["Segment", "Timestamp"])
    grouped = {seg: g for seg, g in hist.groupby("Segment", sort=False)}

    for idx, row in rows.iterrows():
        seg = row["Segment"]
        g = grouped.get(seg)
        if g is None or g.empty:
            continue

        loads = pd.to_numeric(g["Load"], errors="coerce").to_numpy(dtype=float)
        speeds = pd.to_numeric(g["SpeedKmh"], errors="coerce").to_numpy(dtype=float)
        staus = pd.to_numeric(g["Stau_Level"], errors="coerce").to_numpy(dtype=float)
        congs = pd.to_numeric(g["CongestionPercent"], errors="coerce").to_numpy(dtype=float)

        for lag in lags:
            if len(loads) >= lag:
                rows.at[idx, f"Load_lag_{lag}"] = loads[-lag]
            if len(speeds) >= lag:
                rows.at[idx, f"SpeedKmh_lag_{lag}"] = speeds[-lag]
            if len(staus) >= lag:
                rows.at[idx, f"Stau_Level_lag_{lag}"] = staus[-lag]
            if len(congs) >= lag:
                rows.at[idx, f"CongestionPercent_lag_{lag}"] = congs[-lag]

        for window in rolling_windows:
            recent_loads = loads[-window:]
            if len(recent_loads) > 0:
                rows.at[idx, f"Load_roll_mean_{window}"] = float(np.nanmean(recent_loads))
                rows.at[idx, f"Load_roll_max_{window}"] = float(np.nanmax(recent_loads))
                rows.at[idx, f"Load_roll_std_{window}"] = float(np.nanstd(recent_loads, ddof=1)) if len(recent_loads) > 1 else 0.0

            recent_speeds = speeds[-window:]
            if len(recent_speeds) > 0:
                rows.at[idx, f"SpeedKmh_roll_mean_{window}"] = float(np.nanmean(recent_speeds))
                rows.at[idx, f"SpeedKmh_roll_min_{window}"] = float(np.nanmin(recent_speeds))

    rows["LoadToCapacity_lag_1"] = rows.get("Load_lag_1", np.nan) / rows["FlowCapacity"].replace(0, np.nan)
    return rows


def stau_level_from_speed(speed: pd.Series, base_speed: pd.Series) -> pd.Series:
    ratio = pd.to_numeric(speed, errors="coerce") / pd.to_numeric(base_speed, errors="coerce").replace(0, np.nan)
    ratio = ratio.fillna(1.0)
    return pd.Series(
        np.select([ratio >= 0.75, ratio >= 0.50, ratio >= 0.25], [1, 2, 3], default=4),
        index=speed.index,
    ).astype("int64")


def date_range_minutes(start: pd.Timestamp, end: pd.Timestamp, step_minutes: int) -> list[pd.Timestamp]:
    if end < start:
        raise ValueError("Forecast end must be after forecast start.")
    return list(pd.date_range(start=start, end=end, freq=f"{step_minutes}min"))


def finalize_simulation_schema(rows: pd.DataFrame) -> pd.DataFrame:
    out = rows.copy()

    predicted_load = pd.to_numeric(out["Load"], errors="coerce").fillna(0.0).clip(lower=0.0).round().astype("int64")
    out["Anzahl_Autos"] = predicted_load
    out["Load"] = predicted_load

    base_speed = pd.to_numeric(out["BaseSpeed_kmh"], errors="coerce").fillna(30.0).clip(lower=1.0)
    speed = pd.to_numeric(out["SpeedKmh"], errors="coerce").fillna(base_speed)
    speed = speed.clip(lower=3.0, upper=base_speed * 1.10).round(1)
    out["Durchschnittsgeschwindigkeit_kmh"] = speed
    out["SpeedKmh"] = speed

    out["EffectiveCapacity"] = pd.to_numeric(out["EffectiveCapacity"], errors="coerce").fillna(out["FlowCapacity"]).clip(lower=10.0).round(1)
    ratio = predicted_load.astype(float) / out["EffectiveCapacity"].replace(0, np.nan)
    out["CapacityRatio"] = ratio.round(4)
    out["CongestionPercent"] = np.round(np.clip(ratio, 0.0, 2.0) * 100.0, 1)
    out["Stau_Level"] = stau_level_from_speed(out["SpeedKmh"], out["BaseSpeed_kmh"])
    out["Congestion"] = (out["Stau_Level"] >= 3).astype("int64")

    # Enforce the simulator's final column order and dtypes.
    return normalize_dtypes(out[SIMULATION_OUTPUT_COLS])


def forecast_recursive(
    models: ForecastModels,
    prepared_history: pd.DataFrame,
    start: Optional[str],
    until: Optional[str],
    horizon_minutes: int,
    segments: Optional[list[str]],
) -> pd.DataFrame:
    """Starting from the last historic datapoint, continually appends 
    new predicted datapoints and then uses them as part of the input for the next.
    The parameter "start" does not change that and simply controls, at what point the return starts.
    """
    last_ts = pd.Timestamp(prepared_history["Timestamp"].max())
    first_future_ts = last_ts + pd.Timedelta(minutes=models.freq_minutes)

    requested_start = pd.Timestamp(start) if start else first_future_ts
    if requested_start < first_future_ts:
        raise ValueError(
            f"Forecast start {requested_start} is inside the known history. Use a time after {last_ts}."
        )

    requested_end = pd.Timestamp(until) if until else requested_start + pd.Timedelta(minutes=horizon_minutes - models.freq_minutes)
    if requested_end < requested_start:
        raise ValueError("--until must not be before --forecast-start.")
    #what times need to be computed. Called "internal" because depending on the requested_start, not all might be returned, making them thus "external"
    internal_times = date_range_minutes(first_future_ts, requested_end, models.freq_minutes)
    static_df = static_segment_frame(prepared_history)
    #Optionally only forecasts selected segments. Interessting for debugging/ quality control
    if segments:
        static_df = static_df[static_df["Segment"].isin(segments)].copy()
        if static_df.empty:
            raise ValueError("None of the requested --segments were found in the history table.")
        history_state = prepared_history[prepared_history["Segment"].isin(segments)].copy()
    else:
        history_state = prepared_history.copy()

    predictions: list[pd.DataFrame] = []
    #for every timestamp to be predicted: create entries for every segment, add unto them the predicted values.
    for ts in internal_times:
        rows = make_future_base_rows(static_df, ts, models.history_start_ts)
        rows = add_recursive_features_for_step(rows, history_state, models.lags, models.rolling_windows)

        pred_load = np.clip(models.load_model.predict(rows[models.feature_cols]), 0, None)
        pred_speed = np.clip(models.speed_model.predict(rows[models.feature_cols]), 0, None)
        rows["Load"] = pred_load
        rows["SpeedKmh"] = pred_speed

        completed = finalize_simulation_schema(rows)
        predictions.append(completed)

        history_cols = sorted(set(history_state.columns).union(completed.columns))
        history_state = pd.concat(
            [history_state.reindex(columns=history_cols), completed.reindex(columns=history_cols)],
            ignore_index=True,
        )

    out = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame(columns=SIMULATION_OUTPUT_COLS)
    out = out[(out["Timestamp"] >= requested_start) & (out["Timestamp"] <= requested_end)].copy()
    return out[SIMULATION_OUTPUT_COLS].reset_index(drop=True)


def write_forecast_db(
    output_db: str | Path,
    forecasts: pd.DataFrame,
    table: str = TABLE_NAME,
    write_chunk_size: int = 50_000,
) -> None:
    output_db = Path(output_db)
    output_db.parent.mkdir(parents=True, exist_ok=True)

    with connect(output_db) as conn:
        if forecasts.empty:
            forecasts.to_sql(table, conn, if_exists="replace", index=False)
        else:
            first = True
            for start in range(0, len(forecasts), write_chunk_size):
                chunk = forecasts.iloc[start : start + write_chunk_size]
                chunk.to_sql(table, conn, if_exists="replace" if first else "append", index=False)
                first = False

        conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_time ON {table}(Timestamp)")
        conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_segment ON {table}(Segment)")
        conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_datetime_segment ON {table}(Date, Time, Segment)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train XGBoost on the exact simul_data.py traffic_darmstadt schema and write the same schema as forecast output."
    )
    parser.add_argument("--db", default=INPUT_DB, help="Input SQLite database path.")
    parser.add_argument("--table", default=TABLE_NAME, help="Input table name. Defaults to traffic_darmstadt.")
    parser.add_argument("--output-db", default=OUTPUT_DB, help="SQLite DB to write forecasts into.")
    parser.add_argument("--output-table", default=TABLE_NAME, help="Output table name. Defaults to traffic_darmstadt.")

    parser.add_argument("--forecast-start", default=None, help='First forecast timestamp, e.g. "2026-06-23 10:00".')
    parser.add_argument("--until", default=None, help='Last forecast timestamp, e.g. "2026-06-23 17:00".')
    parser.add_argument("--horizon-minutes", type=int, default=60 * 8, help="Forecast horizon if --until is omitted.")

    parser.add_argument("--history-start", default=None, help="Optional lower bound for training history.")
    parser.add_argument("--history-end", default=None, help="Optional upper bound for training history.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Chronological holdout fraction.")
    parser.add_argument("--max-train-rows", type=int, default=None, help="Optional row cap for faster training.")
    parser.add_argument("--max-test-rows", type=int, default=None, help="Optional row cap for faster metric calculation.")

    parser.add_argument("--lags", default=",".join(map(str, DEFAULT_LAGS)), help="Comma-separated lag steps.")
    parser.add_argument("--rolling-windows", default=",".join(map(str, DEFAULT_ROLLING_WINDOWS)), help="Comma-separated rolling windows.")
    parser.add_argument("--segments", default=None, help="Optional comma-separated Segment IDs to forecast/output.")
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    raw = read_history(args.db, args.table)
    #If time is to be restricted, does nothing otherwise:
    raw = filter_history_by_time(raw, args.history_start, args.history_end)

    prepared, lags, windows, freq_minutes = prepare_history(
        raw=raw,
        requested_lags=parse_int_list(args.lags),
        requested_windows=parse_int_list(args.rolling_windows),
    )

    print(f"Loaded {len(raw):,} rows from {args.db} table '{args.table}'.")
    print(f"Found {prepared['Segment'].nunique():,} segments and {prepared['Timestamp'].nunique():,} timestamps.")
    print(f"Inferred step size: {freq_minutes} minute(s). Using lags={lags}, rolling_windows={windows}.")

    models, metrics = train_models(
        prepared=prepared,
        lags=lags,
        windows=windows,
        freq_minutes=freq_minutes,
        test_size=args.test_size,
        random_state=args.random_state,
        max_train_rows=args.max_train_rows,
        max_test_rows=args.max_test_rows,
    )

    print("Holdout metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    forecasts = forecast_recursive(
        models=models,
        prepared_history=prepared,
        start=args.forecast_start,
        until=args.until,
        horizon_minutes=args.horizon_minutes,
        segments=parse_str_list(args.segments),
    )

    write_forecast_db(args.output_db, forecasts, table=args.output_table)
    print(f"Wrote {len(forecasts):,} rows to {args.output_db} table '{args.output_table}'.")
    print("Output schema matches simul_data.py / pg3_forecast.py / pg5_traffic_lights_AI.py expectations.")


if __name__ == "__main__":
    main()
