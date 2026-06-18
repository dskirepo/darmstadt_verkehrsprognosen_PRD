import sqlite3
import time
import numpy as np
import osmnx as ox
import pandas as pd
import pydeck as pdk
import streamlit as st


# ─────────────────────────────────────────────────────────────────────────────
# page setup
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .block-container, .stMainBlockContainer {
        max-width: 94% !important;
        padding-left: 1.2rem !important;
        padding-right: 1.2rem !important;
        padding-top: 1rem !important;
    }
    .small-muted { color:#6b7280; font-size:0.86rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    "<h1 style='text-align: center;'>Prognosebasierte Ampel-Intervention</h1>",
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class='small-muted' style='text-align:center;margin-bottom:1rem;'>
    Liest Verkehrsvorhersagen, erkennt zukünftig überlastete Straßenabschnitte und simuliert eine
    Ampelpriorisierung: belastete Abschnitte bekommen mehr Grünzeit, direkt benachbarte Abschnitte
    tragen dafür zusätzliche Rotlicht-/Wartebelastung.
    </div>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# constants
# ─────────────────────────────────────────────────────────────────────────────
PHASES = ["red", "green", "yellow"]
PHASE_COLORS = {
    "red": [210, 20, 20, 240],
    "yellow": [255, 200, 0, 240],
    "green": [0, 204, 68, 240],
}
PHASE_LABELS = {"red": "Rot", "yellow": "Gelb", "green": "Grün"}

# Dot colors for the intervention view
# Unchanged traffic lights are intentionally neutral/grey; only intersections affected by the forecast intervention receive a signal color
INTERVENTION_DOT_COLORS = {
    "unchanged": [135, 135, 135, 170],
    "green_changed": [0, 204, 68, 245],
    "red_changed": [210, 20, 20, 245],
}
INTERVENTION_DOT_LABELS = {
    "unchanged": "Unverändert",
    "green_changed": "Mehr Grünzeit",
    "red_changed": "Mehr Rot-/Wartezeit",
}


DEFAULT_FORECAST_DB = "traffic_forecasts.db"
DEFAULT_FORECAST_TABLE = "traffic_darmstadt"


# table schema as in data simulation
FORECAST_SCHEMA_COLUMNS = [
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


FORECAST_REQUIRED_COLUMNS = [
    "Timestamp",
    "Segment",
    "u",
    "v",
    "key",
    "Strassenname",
    "Load",
    "EffectiveCapacity",
    "CongestionPercent",
]

# Final-version intervention parameters
INTERVENTION_THRESHOLD_PERCENT = 100.0
INTERVENTION_MAX_TARGETS = 8
INTERVENTION_MAX_RELIEF_FRACTION = 0.28
INTERVENTION_BASE_RELIEF_FRACTION = 0.10
INTERVENTION_NEIGHBOR_BURDEN_FACTOR = 0.85
INTERVENTION_CORRIDOR_DEPTH = 1
INTERVENTION_PROTECT_NEIGHBORS = True

# Keep old manual/demo controls disabled in the final page.
SHOW_MANUAL_TRAFFIC_LIGHT_CONTROLS = False
SHOW_DEBUG_TABLES_AND_EXPORT = False


# ─────────────────────────────────────────────────────────────────────────────
# Helper-functions
# ─────────────────────────────────────────────────────────────────────────────

def normalize_osm_id(value) -> str:
    """Create stable string IDs for OSM u/v/key values loaded from SQLite."""
    if pd.isna(value):
        return "unknown"
    try:
        number = float(value)
        if np.isfinite(number) and abs(number - round(number)) < 1e-9:
            return str(int(round(number)))
    except Exception:
        pass
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def make_segment_id(u, v, key) -> str:
    return f"{normalize_osm_id(u)}_{normalize_osm_id(v)}_{normalize_osm_id(key)}"


def path_from_geometry(geom, fallback_u=None, fallback_v=None, graph=None):
    """Return one or more lon/lat paths for a geometry or a straight fallback line."""
    paths = []
    if geom is not None:
        if geom.geom_type == "LineString":
            paths = [[[float(x), float(y)] for x, y in geom.coords]]
        elif geom.geom_type == "MultiLineString":
            paths = [
                [[float(x), float(y)] for x, y in line.coords]
                for line in geom.geoms
            ]
    if not paths and graph is not None and fallback_u in graph.nodes and fallback_v in graph.nodes:
        u_node = graph.nodes[fallback_u]
        v_node = graph.nodes[fallback_v]
        paths = [[
            [float(u_node["x"]), float(u_node["y"])],
            [float(v_node["x"]), float(v_node["y"])],
        ]]
    return paths


def congestion_color(percent: float) -> list[int]:
    """Traffic coloring for adjusted congestion percent."""
    if pd.isna(percent):
        return [125, 125, 125, 130]
    p = float(percent)
    if p < 55:
        return [  0, 204,  68, 235] # green
    if p < 85:
        return [255, 204,   0, 235] # yellow
    if p < 105:
        return [210,  20,  20, 235] # red
    return [  0,   0,   0, 255] # black


def load_change_color(delta_load: float) -> list[int]:
    """Coloring for the load-change map mode.

    Green means the intervention reduced the load on this segment.
    Red means the intervention increased the load on this segment.
    Grey means no meaningful change.
    """
    try:
        d = float(delta_load)
    except Exception:
        return [135, 135, 135, 110] #grey

    if abs(d) < 0.5:
        return [135, 135, 135, 110]
    if d < 0:
        return [0, 180, 80, 230]
    return [215, 30, 30, 230]


def load_change_label(delta_load: float) -> str:
    try:
        d = float(delta_load)
    except Exception:
        return "unverändert"
    if abs(d) < 0.5:
        return "unverändert"
    return "Last gesenkt" if d < 0 else "Last erhöht"


def safe_float(series_or_value, fallback=0.0):
    try:
        return float(series_or_value)
    except Exception:
        return fallback


# ─────────────────────────────────────────────────────────────────────────────
# OSM data
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Straßennetz wird geladen …")
def load_all_data():
    G = ox.graph_from_place("Darmstadt-Mitte, Darmstadt, Germany", network_type="drive")
    nodes_gdf, edges_gdf = ox.graph_to_gdfs(G)
    edges_gdf = edges_gdf.reset_index()

    # intersections: nodes with at least 3 connected directed edges
    intersection_rows = []
    for node_id in G.nodes():
        if G.degree(node_id) < 3:
            continue
        nd = G.nodes[node_id]
        streets: set[str] = set()
        for _, _, ed in G.edges(node_id, data=True):
            name = ed.get("name", "")
            if isinstance(name, list):
                streets.update(str(n) for n in name if n)
            elif name:
                streets.add(str(name))
        label = " / ".join(sorted(streets)[:2]) if streets else f"Knotenpunkt {node_id}"
        intersection_rows.append(
            {
                "node_id": int(node_id),
                "lat": float(nd["y"]),
                "lon": float(nd["x"]),
                "label": label,
            }
        )

    df_nodes = pd.DataFrame(intersection_rows)
    if not df_nodes.empty:
        label_cnt = df_nodes["label"].value_counts()
        df_nodes["unique_label"] = df_nodes.apply(
            lambda r: f"{r['label']} (#{r['node_id']})"
            if label_cnt.get(r["label"], 0) > 1
            else r["label"],
            axis=1,
        )
        df_nodes = df_nodes.sort_values("unique_label").reset_index(drop=True)

    edge_path_rows = []
    edge_static_rows = []
    for _, row in edges_gdf.iterrows():
        u = row["u"]
        v = row["v"]
        key = row["key"]
        segment = make_segment_id(u, v, key)
        name = row.get("name", "Unbekannt")
        if isinstance(name, list):
            name = name[0] if name else "Unbekannt"
        highway = row.get("highway", "Unbekannt")
        if isinstance(highway, list):
            highway = highway[0] if highway else "Unbekannt"

        edge_static_rows.append(
            {
                "Segment": segment,
                "u": normalize_osm_id(u),
                "v": normalize_osm_id(v),
                "key": normalize_osm_id(key),
                "Street_osm": str(name),
                "Highway_osm": str(highway),
            }
        )

        for path in path_from_geometry(row.geometry, u, v, G):
            edge_path_rows.append(
                {
                    "Segment": segment,
                    "u": normalize_osm_id(u),
                    "v": normalize_osm_id(v),
                    "key": normalize_osm_id(key),
                    "Street_osm": str(name),
                    "Highway_osm": str(highway),
                    "path": path,
                }
            )

    edge_paths_df = pd.DataFrame(edge_path_rows)
    edge_static_df = pd.DataFrame(edge_static_rows).drop_duplicates("Segment")

    # Neighbor lookup: all directly connected directed segments sharing either endpoint (traffic-light trade-offs at nearby intersections)
    node_to_segments: dict[str, set[str]] = {}
    for _, row in edge_static_df.iterrows():
        node_to_segments.setdefault(row["u"], set()).add(row["Segment"])
        node_to_segments.setdefault(row["v"], set()).add(row["Segment"])

    neighbor_lookup: dict[str, list[str]] = {}
    endpoint_lookup: dict[str, tuple[str, str]] = {}
    for _, row in edge_static_df.iterrows():
        seg = row["Segment"]
        u = row["u"]
        v = row["v"]
        endpoint_lookup[seg] = (u, v)
        neighbors = set(node_to_segments.get(u, set())) | set(node_to_segments.get(v, set()))
        neighbors.discard(seg)
        neighbor_lookup[seg] = sorted(neighbors)

    bounds = edges_gdf.total_bounds.tolist()
    return df_nodes, edge_paths_df, edge_static_df, neighbor_lookup, endpoint_lookup, bounds


intersections, edge_paths_df, edge_static_df, neighbor_lookup, endpoint_lookup, bounds = load_all_data()

if intersections.empty:
    st.error("Keine Kreuzungen gefunden.")
    st.stop()

centre_lon = (bounds[0] + bounds[2]) / 2
centre_lat = (bounds[1] + bounds[3]) / 2


# ─────────────────────────────────────────────────────────────────────────────
# forecast loading / normalization
# ─────────────────────────────────────────────────────────────────────────────

def validate_forecast_schema(raw: pd.DataFrame) -> None:
    """Validate the pg2-compatible forecast table written by the forecast script."""
    missing_required = [c for c in FORECAST_REQUIRED_COLUMNS if c not in raw.columns]
    if missing_required:
        raise ValueError(
            "Die Forecast-DB passt nicht zum erwarteten Schema. "
            f"Erwartet wird die Tabelle '{DEFAULT_FORECAST_TABLE}' aus "
            "xgb_forecast.py. "
            f"Fehlende Pflichtspalten: {missing_required}"
        )


@st.cache_data(show_spinner="Prognose wird aus SQLite geladen …")
def load_traffic_data() -> pd.DataFrame:
    """Read forecasts exactly like pg2: traffic_forecasts.db -> traffic_darmstadt."""
    conn = sqlite3.connect(DEFAULT_FORECAST_DB)
    try:
        raw = pd.read_sql_query(f'SELECT * FROM {DEFAULT_FORECAST_TABLE}', conn)
    finally:
        conn.close()

    validate_forecast_schema(raw)
    raw["Timestamp"] = pd.to_datetime(raw["Timestamp"])
    return normalize_forecast_dataframe(raw)

def normalize_forecast_dataframe(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return raw.copy()

    df = raw.copy()

    # Timestamp
    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    elif {"Date", "Time"}.issubset(df.columns):
        df["Timestamp"] = pd.to_datetime(df["Date"].astype(str) + " " + df["Time"].astype(str), errors="coerce")
    else:
        raise ValueError("Die Tabelle enthält weder Timestamp noch Date+Time.")
    df = df[df["Timestamp"].notna()].copy()

    # Segment identity
    for col in ["u", "v", "key", "FromNode", "ToNode"]:
        if col in df.columns:
            df[col] = df[col].map(normalize_osm_id)

    if "Segment" not in df.columns:
        if {"u", "v", "key"}.issubset(df.columns):
            df["Segment"] = [make_segment_id(u, v, k) for u, v, k in zip(df["u"], df["v"], df["key"])]
        elif {"FromNode", "ToNode"}.issubset(df.columns):
            df["Segment"] = [make_segment_id(u, v, 0) for u, v in zip(df["FromNode"], df["ToNode"])]
        else:
            raise ValueError("Die Tabelle enthält keine nutzbare Segment-ID.")
    else:
        df["Segment"] = df["Segment"].astype(str)

    if "u" not in df.columns or "v" not in df.columns or "key" not in df.columns:
        split = df["Segment"].astype(str).str.split("_", n=2, expand=True)
        if split.shape[1] >= 3:
            df["u"] = split[0].map(normalize_osm_id)
            df["v"] = split[1].map(normalize_osm_id)
            df["key"] = split[2].map(normalize_osm_id)

    # Load / forecast values
    if "PredictedLoad" in df.columns:
        df["ForecastLoad"] = pd.to_numeric(df["PredictedLoad"], errors="coerce")
    elif "Load" in df.columns:
        df["ForecastLoad"] = pd.to_numeric(df["Load"], errors="coerce")
    elif "Anzahl_Autos" in df.columns:
        df["ForecastLoad"] = pd.to_numeric(df["Anzahl_Autos"], errors="coerce")
    else:
        raise ValueError("Keine Last-Spalte gefunden: erwartet PredictedLoad, Load oder Anzahl_Autos.")
    df["ForecastLoad"] = df["ForecastLoad"].fillna(0.0).clip(lower=0.0)

    # Capacity
    if "EffectiveCapacity" in df.columns:
        df["Capacity"] = pd.to_numeric(df["EffectiveCapacity"], errors="coerce")
    elif "FlowCapacity" in df.columns:
        df["Capacity"] = pd.to_numeric(df["FlowCapacity"], errors="coerce")
    elif "StorageCapacity" in df.columns:
        df["Capacity"] = pd.to_numeric(df["StorageCapacity"], errors="coerce")
    else:
        df["Capacity"] = df.groupby("Segment")["ForecastLoad"].transform(lambda s: max(1.0, s.quantile(0.95) * 1.10))
    df["Capacity"] = df["Capacity"].fillna(df.groupby("Segment")["ForecastLoad"].transform("max") * 1.10)
    df["Capacity"] = df["Capacity"].fillna(1.0).clip(lower=1.0)

    # Congestion percent
    if "PredictedCongestionPercent" in df.columns:
        df["ForecastCongestionPercent"] = pd.to_numeric(df["PredictedCongestionPercent"], errors="coerce")
    elif "CongestionPercent" in df.columns:
        df["ForecastCongestionPercent"] = pd.to_numeric(df["CongestionPercent"], errors="coerce")
    else:
        df["ForecastCongestionPercent"] = 100.0 * df["ForecastLoad"] / df["Capacity"].replace(0, np.nan)
    df["ForecastCongestionPercent"] = df["ForecastCongestionPercent"].fillna(
        100.0 * df["ForecastLoad"] / df["Capacity"].replace(0, np.nan)
    )

    if "PredictedSpeedKmh" in df.columns:
        df["ForecastSpeedKmh"] = pd.to_numeric(df["PredictedSpeedKmh"], errors="coerce")
    elif "Durchschnittsgeschwindigkeit_kmh" in df.columns:
        df["ForecastSpeedKmh"] = pd.to_numeric(df["Durchschnittsgeschwindigkeit_kmh"], errors="coerce")
    elif "SpeedKmh" in df.columns:
        df["ForecastSpeedKmh"] = pd.to_numeric(df["SpeedKmh"], errors="coerce")
    else:
        df["ForecastSpeedKmh"] = np.nan

    if "PredictedStauLevel" in df.columns:
        df["ForecastStauLevel"] = pd.to_numeric(df["PredictedStauLevel"], errors="coerce")
    elif "Stau_Level" in df.columns:
        df["ForecastStauLevel"] = pd.to_numeric(df["Stau_Level"], errors="coerce")
    elif "StauLevel" in df.columns:
        df["ForecastStauLevel"] = pd.to_numeric(df["StauLevel"], errors="coerce")
    else:
        df["ForecastStauLevel"] = np.select(
            [df["ForecastCongestionPercent"] < 60, df["ForecastCongestionPercent"] < 100],
            [1, 2],
            default=3,
        )

    if "Street" not in df.columns:
        if "Strassenname" in df.columns:
            df["Street"] = df["Strassenname"].fillna("Unknown").astype(str)
        else:
            df["Street"] = df["Segment"].astype(str)
    if "Strassenname" not in df.columns:
        df["Strassenname"] = df["Street"].fillna("Unknown").astype(str)

    if "Highway" not in df.columns:
        df["Highway"] = "unknown"

    keep = [
        "Timestamp", "Segment", "u", "v", "key", "Street", "Strassenname", "Highway",
        "ForecastLoad", "Capacity", "ForecastCongestionPercent", "ForecastSpeedKmh",
        "ForecastStauLevel",
    ]
    for optional in [
        "FlowCapacity", "StorageCapacity", "EffectiveCapacity", "Length_Meter", "Lanes",
        "RoadImportance", "BaseSpeed_kmh", "CongestionProbability", "PredictedCongested",
        "ModelCreatedAt", "Date", "Time",
    ]:
        if optional in df.columns and optional not in keep:
            keep.append(optional)

    out = df[[c for c in keep if c in df.columns]].copy()
    out["ForecastTable"] = DEFAULT_FORECAST_TABLE
    return out.sort_values(["Timestamp", "Segment"]).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# REDIRECTION / SIGNAL-CONTROL MODEL
# ─────────────────────────────────────────────────────────────────────────────
def same_street_corridor(
    segment: str,
    frame: pd.DataFrame,
    neighbor_lookup: dict[str, list[str]],
    depth: int = 1,
) -> set[str]:
    """Find directly connected same-street segments for a small green wave."""
    if segment not in set(frame["Segment"]):
        return {segment}

    street_by_segment = frame.set_index("Segment")["Street"].astype(str).to_dict()
    base_street = street_by_segment.get(segment, "")
    if not base_street or base_street.lower() in {"unknown", "none", "nan"}:
        return {segment}

    seen = {segment}
    frontier = {segment}
    for _ in range(max(0, depth)):
        next_frontier = set()
        for seg in frontier:
            for nb in neighbor_lookup.get(seg, []):
                if nb in seen:
                    continue
                if street_by_segment.get(nb, "") == base_street:
                    seen.add(nb)
                    next_frontier.add(nb)
        frontier = next_frontier
        if not frontier:
            break
    return seen


def apply_signal_intervention(
    current: pd.DataFrame,
    neighbor_lookup: dict[str, list[str]],
    threshold_percent: float = 100.0,
    max_targets: int = 8,
    max_relief_fraction: float = 0.28,
    base_relief_fraction: float = 0.10,
    neighbor_burden_factor: float = 0.85,
    corridor_depth: int = 1,
    protect_already_congested_neighbors: bool = True,
) -> pd.DataFrame:
    """
    A deliberately simple, explainable control heuristic.

    For each predicted-congested target segment:
    - mark the target/corridor as green-priority
    - reduce its adjusted load to represent more green time
    - distribute a red-light burden to directly connected neighboring segments

    This is not a real traffic-engineering optimizer. It is a visual/plausible
    proof-of-concept that creates data a later optimizer could replace.
    """
    df = current.copy().reset_index(drop=True)
    if df.empty:
        return df

    df["AdjustedLoad"] = (
        pd.to_numeric(df["ForecastLoad"], errors="coerce")
        .fillna(0.0)
        .clip(lower=0.0)
        .astype("float64")
    )
    df["AdjustedCapacity"] = (
        pd.to_numeric(df["Capacity"], errors="coerce")
        .fillna(1.0)
        .clip(lower=1.0)
        .astype("float64")
    )
    df["DeltaLoad"] = np.zeros(len(df), dtype="float64")
    df["RelievedLoad"] = np.zeros(len(df), dtype="float64")
    df["AddedNeighborBurden"] = np.zeros(len(df), dtype="float64")
    df["GreenPriority"] = 0
    df["RedBurden"] = 0
    df["InterventionReason"] = ""

    segment_to_idx = {seg: idx for idx, seg in enumerate(df["Segment"].astype(str))}

    candidates = df[df["ForecastCongestionPercent"] >= threshold_percent].copy()
    if candidates.empty:
        df["AdjustedCongestionPercent"] = 100.0 * df["AdjustedLoad"] / df["AdjustedCapacity"].replace(0, np.nan)
        df["InterventionRole"] = "none"
        return df

    candidates["Excess"] = candidates["ForecastCongestionPercent"] - threshold_percent
    candidates = candidates.sort_values("Excess", ascending=False).head(max_targets)

    for _, target_row in candidates.iterrows():
        target = str(target_row["Segment"])
        if target not in segment_to_idx:
            continue

        # The target segment and a short same-street corridor receive the green priority
        corridor = same_street_corridor(target, df, neighbor_lookup, depth=corridor_depth)
        corridor = {seg for seg in corridor if seg in segment_to_idx}
        if not corridor:
            corridor = {target}

        target_idx = segment_to_idx[target]
        target_street = df.at[target_idx, "Street"]
        target_name = target_street if target_street and target_street != "{Unbekannte Straße}" else f"Segment {target}"
        severity = max(0.0, float(df.at[target_idx, "ForecastCongestionPercent"] - threshold_percent) / max(1.0, threshold_percent))
        relief_fraction = min(max_relief_fraction, base_relief_fraction + 0.16 * severity)

        # Burden neighbors are directly connected to the corridor but are not part of it
        burden_neighbors = set()
        for seg in corridor:
            burden_neighbors.update(neighbor_lookup.get(seg, []))
        burden_neighbors = {seg for seg in burden_neighbors if seg in segment_to_idx and seg not in corridor}

        if protect_already_congested_neighbors:
            # Prefer neighbors that still have some spare capacity; if all direct neighbors are congested, keep them anyway so the trade-off remains visible
            less_bad = {
                seg for seg in burden_neighbors
                if df.at[segment_to_idx[seg], "ForecastCongestionPercent"] < threshold_percent
            }
            if less_bad:
                burden_neighbors = less_bad

        # Reduce corridor loads; main target gets full relief, same-street corridor pieces get half relief
        total_relieved = 0.0
        for seg in corridor:
            idx = segment_to_idx[seg]
            factor = relief_fraction if seg == target else relief_fraction * 0.50
            relieved = float(df.at[idx, "AdjustedLoad"]) * factor
            if relieved <= 0:
                continue
            df.at[idx, "AdjustedLoad"] = max(0.0, float(df.at[idx, "AdjustedLoad"]) - relieved)
            df.at[idx, "DeltaLoad"] -= relieved
            df.at[idx, "RelievedLoad"] += relieved
            df.at[idx, "GreenPriority"] = 1
            msg = f"Grünpriorität wegen {target_name}, {target}"
            df.at[idx, "InterventionReason"] = (df.at[idx, "InterventionReason"] + "; " + msg).strip("; ")
            total_relieved += relieved

        if total_relieved <= 0 or not burden_neighbors:
            continue

        total_added = total_relieved * neighbor_burden_factor

        # use spare capacity as weights; if no spare capacity exists, distribute evenly
        weights = []
        neighbors_sorted = sorted(burden_neighbors)
        for seg in neighbors_sorted:
            idx = segment_to_idx[seg]
            spare = max(0.0, float(df.at[idx, "AdjustedCapacity"] - df.at[idx, "AdjustedLoad"]))
            weights.append(spare)
        weights_arr = np.asarray(weights, dtype=float)
        if weights_arr.sum() <= 0:
            weights_arr = np.ones(len(neighbors_sorted), dtype=float)
        weights_arr = weights_arr / weights_arr.sum()

        for seg, weight in zip(neighbors_sorted, weights_arr):
            idx = segment_to_idx[seg]
            added = float(total_added * weight)
            df.at[idx, "AdjustedLoad"] += added
            df.at[idx, "DeltaLoad"] += added
            df.at[idx, "AddedNeighborBurden"] += added
            df.at[idx, "RedBurden"] = 1
            msg = f"Rot-/Wartebelastung wegen Priorität {target_name}, {target}"
            df.at[idx, "InterventionReason"] = (df.at[idx, "InterventionReason"] + "; " + msg).strip("; ")

    df["AdjustedCongestionPercent"] = 100.0 * df["AdjustedLoad"] / df["AdjustedCapacity"].replace(0, np.nan)
    df["CongestionPrevented"] = (
        (df["ForecastCongestionPercent"] >= threshold_percent)
        & (df["AdjustedCongestionPercent"] < threshold_percent)
    ).astype(int)

    def role(row) -> str:
        if int(row.get("GreenPriority", 0)) == 1 and int(row.get("RedBurden", 0)) == 1:
            return "Grünpriorität & Rotbelastung"
        if int(row.get("GreenPriority", 0)) == 1:
            return "Grünpriorität"
        if int(row.get("RedBurden", 0)) == 1:
            return "Rotbelastung"
        return "keine"

    df["InterventionRole"] = df.apply(role, axis=1)
    return df


def intervention_nodes(result: pd.DataFrame) -> tuple[set[str], set[str]]:
    green_nodes: set[str] = set()
    red_nodes: set[str] = set()
    if result.empty:
        return green_nodes, red_nodes

    for _, row in result.iterrows():
        seg = str(row["Segment"])
        endpoints = endpoint_lookup.get(seg)
        if not endpoints:
            continue
        if int(row.get("GreenPriority", 0)) == 1:
            green_nodes.update(endpoints)
        elif int(row.get("RedBurden", 0)) == 1:
            red_nodes.update(endpoints)

    # green priority dominates the single-dot visualization if both occur at a node
    red_nodes = red_nodes - green_nodes
    return green_nodes, red_nodes


def build_display_light_states(
    base_states: dict[int, str],
    green_nodes: set[str],
    red_nodes: set[str],
    node_ids: list[int],
) -> dict[int, str]:
    out = {int(k): str(v) for k, v in base_states.items()}
    for nid in node_ids:
        nid_s = str(int(nid))
        if nid_s in green_nodes:
            out[int(nid)] = "green"
        elif nid_s in red_nodes:
            out[int(nid)] = "red"
    return out


def prepare_path_layers(result: pd.DataFrame) -> pd.DataFrame:
    if result.empty:
        return pd.DataFrame()
    merged = edge_paths_df.merge(result, on="Segment", how="inner", suffixes=("_osm", ""))
    if merged.empty:
        return merged

    merged["display_street"] = merged["Street"].fillna(merged.get("Street_osm", "Unknown"))

    # map mode 1: intervention effect
    # green = load decreased, red = load increased, grey = unchanged
    delta = pd.to_numeric(merged.get("DeltaLoad", 0.0), errors="coerce").fillna(0.0)
    max_abs_delta = float(max(1.0, delta.abs().quantile(0.95)))
    merged["load_change_color"] = delta.map(load_change_color)
    merged["load_change_label"] = delta.map(load_change_label)
    merged["load_change_width"] = 2.5 + 8.0 * np.minimum(delta.abs() / max_abs_delta, 1.0)
    merged.loc[delta.abs() < 0.5, "load_change_width"] = 2.2

    # Map mode 2: congestion after intervention
    merged["congestion_map_color"] = merged["AdjustedCongestionPercent"].map(congestion_color)
    merged["congestion_map_width"] = np.select(
        [
            merged["AdjustedCongestionPercent"].ge(105),
            merged["AdjustedCongestionPercent"].ge(85),
            merged["AdjustedCongestionPercent"].ge(55),
        ],
        [8.5, 6.5, 5.0],
        default=3.5,
    )

    merged["tooltip"] = merged.apply(
        lambda r: (
            f"<b>{r.get('display_street', 'Unknown')}</b><br/>"
            f"Prognose: {safe_float(r.get('ForecastLoad')):.0f} Fahrzeuge, "
            f"{safe_float(r.get('ForecastCongestionPercent')):.1f}%<br/>"
            f"Nach Intervention: {safe_float(r.get('AdjustedLoad')):.0f} Fahrzeuge, "
            f"{safe_float(r.get('AdjustedCongestionPercent')):.1f}%<br/>"
            f"Δ Last: {safe_float(r.get('DeltaLoad')):+.1f} "
            f"({r.get('load_change_label', 'unverändert')})<br/>"
            f"Rolle: {r.get('InterventionRole', 'keine')}<br/>"
            f"{r.get('InterventionReason', '')}"
        ),
        axis=1,
    )
    return merged


def export_adjusted_result(db_path: str, result: pd.DataFrame, table_name: str = "traffic_signal_intervention") -> None:
    if not db_path or result.empty:
        return
    export_cols = [
        "Timestamp", "Segment", "Street", "ForecastLoad", "Capacity", "ForecastCongestionPercent",
        "AdjustedLoad", "AdjustedCongestionPercent", "DeltaLoad", "RelievedLoad",
        "AddedNeighborBurden", "GreenPriority", "RedBurden", "CongestionPrevented",
        "InterventionRole", "InterventionReason",
    ]
    frame = result[[c for c in export_cols if c in result.columns]].copy()
    with sqlite3.connect(db_path) as conn:
        frame.to_sql(table_name, conn, if_exists="replace", index=False)

# ─────────────────────────────────────────────────────────────────────────────
# forecast database
# ─────────────────────────────────────────────────────────────────────────────

db_path = DEFAULT_FORECAST_DB
try:
    forecast_df = load_traffic_data()
except Exception as exc:
    st.error(f"Prognose konnte nicht geladen werden: {exc}")
    st.info(
        "Die KI-Ampelsteuerung erwartet dieselbe Forecast-Datenbank wie die Prognose: "
        f"`{DEFAULT_FORECAST_DB}` mit der Tabelle `{DEFAULT_FORECAST_TABLE}`."
    )
    forecast_df = pd.DataFrame()

available_timestamps = sorted(pd.to_datetime(forecast_df["Timestamp"].unique()))
unique_dates = sorted(set(ts.date() for ts in available_timestamps))

# ─────────────────────────────────────────────────────────────────────────────
# session state
# ─────────────────────────────────────────────────────────────────────────────

if "light_states" not in st.session_state:
    rng = np.random.default_rng(42)
    st.session_state.light_states = {
        int(r["node_id"]): PHASES[int(rng.integers(0, 3))]
        for _, r in intersections.iterrows()
    }

for _, _row in intersections.iterrows():
    _nid = int(_row["node_id"])
    if _nid not in st.session_state.light_states:
        st.session_state.light_states[_nid] = "red"

if "selected_node" not in st.session_state:
    st.session_state.selected_node = int(intersections.iloc[0]["node_id"])

if "auto_cycle" not in st.session_state:
    st.session_state.auto_cycle = False

if "use_forecast_control" not in st.session_state:
    st.session_state.use_forecast_control = True

if "pg5_selected_dt" not in st.session_state or st.session_state.pg5_selected_dt not in available_timestamps:
    st.session_state.pg5_selected_dt = available_timestamps[0]

def set_active_date(new_date): # triggered when a date button is clicked.
    first_of_day = next(t for t in available_timestamps if t.date() == new_date)
    st.session_state.pg5_selected_dt = first_of_day

def go_prev(): # triggered by the ◀ button.
    idx = available_timestamps.index(st.session_state.pg5_selected_dt)
    if idx > 0:
        st.session_state.pg5_selected_dt = available_timestamps[idx - 1]

def go_next(): # triggered by the ▶ button.
    idx = available_timestamps.index(st.session_state.pg5_selected_dt)
    if idx < len(available_timestamps) - 1:
        st.session_state.pg5_selected_dt = available_timestamps[idx + 1]


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR / CONTROL INPUTS
# ─────────────────────────────────────────────────────────────────────────────

col_map, col_ctrl = st.columns([3.2, 1.25])

with col_ctrl:
    if not forecast_df.empty:
        if len(unique_dates) == 1:
            st.markdown(
                f"<h3 style='text-align:center; margin-bottom:0px;'>"
                f"{unique_dates[0].strftime('%A, %d. %B %Y')}</h3>",
                unsafe_allow_html=True,
            )
        else:
            date_cols = st.columns(len(unique_dates))
            for i, d in enumerate(unique_dates):
                with date_cols[i]:
                    # Check if this button's date matches the currently selected date
                    is_active = (d == st.session_state.pg5_selected_dt.date())
                    st.button(
                        label=d.strftime("%a %d.%m."),
                        key=f"pg2_date_btn_{i}",
                        use_container_width=True,
                        type="primary" if is_active else "secondary",
                        on_click=set_active_date,
                        args=(d,)  # Pass the target date to the callback
                    )

        ctrl_prev, ctrl_dt, ctrl_next = st.columns([1, 6, 1])
    
        with ctrl_prev:
            st.write("")
            st.button("◀", use_container_width=True, help="Vorherige Stunde", on_click=go_prev)

        with ctrl_next:
            st.write("")
            st.button("▶", use_container_width=True, help="Nächste Stunde", on_click=go_next)

        # select forecast time
        selected_date   = st.session_state.pg5_selected_dt.date()
        day_timestamps  = [t for t in available_timestamps if t.date() == selected_date]

        if st.session_state.pg5_selected_dt not in day_timestamps:
            st.session_state.pg5_selected_dt = day_timestamps[0]

        with ctrl_dt:
            selected_ts = st.selectbox(
                "Uhrzeit",
                key="pg5_selected_dt",   # session state IS the selected value
                options=day_timestamps,  # only this day's timestamps
                format_func=lambda ts: ts.strftime("%H:%M"),
            )
    else:
        selected_ts = None

    st.markdown("<br>", unsafe_allow_html=True)
    st.toggle(
        "Automatische Ampel-Intervention anzeigen",
        key="use_forecast_control",
        help="Wenn aktiv, überschreibt die Prognoselogik die angezeigten Ampelphasen visuell."
    )

    st.markdown("<br>", unsafe_allow_html=True)
    map_mode = st.radio(
        "**Kartenmodus**",
        options=[
            "Laständerung durch Intervention",
            "Staukarte nach Intervention",
        ],
        index=0,
    )

    if map_mode == "Laständerung durch Intervention":
        st.markdown(
            """
            <div style='font-size:0.82rem;line-height:1.7;color:#ddd;background:#1a1a2e;
                        padding:9px 12px;border-radius:8px;margin-bottom:8px;'>
            <b>Kartenlegende</b><br/>
            <span style='color:#00b450;font-size:1.2rem;'>●</span> Segment entlastet<br/>
            <span style='color:#d71e1e;font-size:1.2rem;'>●</span> Segment zusätzlich belastet<br/>
            <span style='color:#888;font-size:1.2rem;'>●</span> keine Laständerung<br/>
            Ampelpunkte: grau = unverändert, grün/rot = durch Intervention verändert
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div style='font-size:0.82rem;line-height:1.7;color:#ddd;background:#1a1a2e;
                        padding:9px 12px;border-radius:8px;margin-bottom:8px;'>
            <b>Kartenlegende</b><br/>
            <span style='color:#00cc44; font-size:1.3rem; vertical-align:middle;'>●</span> geringe Auslastung<br/>
            <span style='color:#ffcc00; font-size:1.3rem; vertical-align:middle;'>●</span> mittlere Auslastung<br/>
            <span style='color:#d21414; font-size:1.3rem; vertical-align:middle;'>●</span> hohe Auslastung<br/>
            <span style='color:#000000; font-size:1.3rem; vertical-align:middle; -webkit-text-stroke: 0.3px white;'>●</span> überlastet / Stau<br/>
            Ampelpunkte: grau = unverändert, grün/rot = durch Intervention verändert
            </div>
            """,
            unsafe_allow_html=True,
        )

    threshold_percent = INTERVENTION_THRESHOLD_PERCENT
    max_targets = INTERVENTION_MAX_TARGETS
    max_relief_fraction = INTERVENTION_MAX_RELIEF_FRACTION
    base_relief_fraction = INTERVENTION_BASE_RELIEF_FRACTION
    neighbor_burden_factor = INTERVENTION_NEIGHBOR_BURDEN_FACTOR
    corridor_depth = INTERVENTION_CORRIDOR_DEPTH
    protect_neighbors = INTERVENTION_PROTECT_NEIGHBORS

    if SHOW_MANUAL_TRAFFIC_LIGHT_CONTROLS:
        st.markdown("### Manuelle Ampeln")
        label_to_id: dict[str, int] = dict(zip(intersections["unique_label"], intersections["node_id"].astype(int)))
        id_to_label: dict[int, str] = {v: k for k, v in label_to_id.items()}
        labels = list(label_to_id.keys())
        def_label = id_to_label.get(st.session_state.selected_node, labels[0])

        chosen_label = st.selectbox(
            "Kreuzung auswählen",
            options=labels,
            index=labels.index(def_label) if def_label in labels else 0,
        )
        sel_id = label_to_id[chosen_label]
        st.session_state.selected_node = sel_id
        cur_phase = st.session_state.light_states.get(sel_id, "red")

        left_col, right_col = st.columns([1, 1])
        with left_col:
            def traffic_light_html(phase: str) -> str:
                off = {"red": "#3a0a0a", "yellow": "#2e2800", "green": "#002e10"}
                on_ = {"red": "#ee1c1c", "yellow": "#ffc800", "green": "#00dd44"}
                glow = {
                    "red": ("filter:drop-shadow(0 0 11px #ff2222cc)", "", ""),
                    "yellow": ("", "filter:drop-shadow(0 0 11px #ffc800cc)", ""),
                    "green": ("", "", "filter:drop-shadow(0 0 11px #00ff44cc)"),
                }
                lights = [
                    (on_["red"] if phase == "red" else off["red"], glow[phase][0]),
                    (on_["yellow"] if phase == "yellow" else off["yellow"], glow[phase][1]),
                    (on_["green"] if phase == "green" else off["green"], glow[phase][2]),
                ]
                circles = "".join(
                    f"<div style='width:25px;height:25px;border-radius:50%;background:{c};"
                    f"margin:3px auto;{g};box-shadow:inset 0 -1px 3px rgba(0,0,0,.4);'></div>"
                    for c, g in lights
                )
                return (
                    "<div style='display:flex;flex-direction:column;align-items:center;"
                    "background:#0e0e1c;border-radius:8px;padding:11px 15px;"
                    "border:2px solid #2a2a3e;width:fit-content;margin:5px auto;'>"
                    "<div style='width:4px;height:8px;background:#555;border-radius:2px 2px 0 0;"
                    "margin-bottom:2px;'></div>"
                    + circles
                    + "</div>"
                )

            st.markdown(traffic_light_html(cur_phase), unsafe_allow_html=True)

        with right_col:
            if st.button("**Rot**", width='stretch', type="primary" if cur_phase == "red" else "secondary"):
                st.session_state.light_states[sel_id] = "red"
                st.rerun()
            if st.button("**Gelb**", width='stretch', type="primary" if cur_phase == "yellow" else "secondary"):
                st.session_state.light_states[sel_id] = "yellow"
                st.rerun()
            if st.button("**Grün**", width='stretch', type="primary" if cur_phase == "green" else "secondary"):
                st.session_state.light_states[sel_id] = "green"
                st.rerun()

        bc1, bc2 = st.columns(2)
        with bc1:
            if st.button("Alle Rot", width='stretch'):
                for k in st.session_state.light_states:
                    st.session_state.light_states[k] = "red"
                st.rerun()
        with bc2:
            if st.button("Alle Grün", width='stretch'):
                for k in st.session_state.light_states:
                    st.session_state.light_states[k] = "green"
                st.rerun()

        auto_lbl = "⏹ Auto-Zyklus" if st.session_state.auto_cycle else "▶ Auto-Zyklus"
        if st.button(auto_lbl, width='stretch'):
            st.session_state.auto_cycle = not st.session_state.auto_cycle
            st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# COMPUTE CURRENT INTERVENTION
# ─────────────────────────────────────────────────────────────────────────────

if selected_ts is not None and not forecast_df.empty:
    current_forecast = forecast_df[forecast_df["Timestamp"].eq(selected_ts)].copy()
    result = apply_signal_intervention(
        current=current_forecast,
        neighbor_lookup=neighbor_lookup,
        threshold_percent=float(threshold_percent),
        max_targets=int(max_targets),
        max_relief_fraction=float(max_relief_fraction),
        base_relief_fraction=float(base_relief_fraction),
        neighbor_burden_factor=float(neighbor_burden_factor),
        corridor_depth=int(corridor_depth),
        protect_already_congested_neighbors=bool(protect_neighbors),
    )
else:
    current_forecast = pd.DataFrame()
    result = pd.DataFrame()

path_data = prepare_path_layers(result)
green_nodes, red_nodes = intervention_nodes(result) if st.session_state.use_forecast_control else (set(), set())
node_ids = [int(x) for x in intersections["node_id"].tolist()]
display_light_states = build_display_light_states(st.session_state.light_states, green_nodes, red_nodes, node_ids)


# ─────────────────────────────────────────────────────────────────────────────
# map
# ─────────────────────────────────────────────────────────────────────────────

with col_map:
    if not forecast_df.empty and result.empty:
        st.info("Für diesen Zeitpunkt liegen keine passenden Forecast-Zeilen vor.")

    matching_ratio = 0.0
    if not result.empty:
        matching_segments = set(path_data["Segment"].unique()) if not path_data.empty else set()
        matching_ratio = len(matching_segments) / max(1, result["Segment"].nunique())
        if matching_ratio < 0.65:
            st.warning(
                f"Nur {matching_ratio:.0%} der Forecast-Segmente konnten dem aktuell geladenen OSM-Netz zugeordnet werden. "
                "Wenn die Forecast-DB mit einer anderen OSM-Version erzeugt wurde, können Kanten-IDs abweichen."
            )

    scatter_data = []
    for _, row in intersections.iterrows():
        nid = int(row["node_id"])
        is_auto_green = st.session_state.use_forecast_control and str(nid) in green_nodes
        is_auto_red = st.session_state.use_forecast_control and str(nid) in red_nodes

        if st.session_state.use_forecast_control:
            if is_auto_green:
                dot_key = "green_changed"
            elif is_auto_red:
                dot_key = "red_changed"
            else:
                dot_key = "unchanged"

            dot_color = INTERVENTION_DOT_COLORS[dot_key]
            dot_label = INTERVENTION_DOT_LABELS[dot_key]
            auto_note = (
                "durch Intervention verändert"
                if dot_key != "unchanged"
                else "nicht durch Intervention verändert"
            )
        else:
            phase = st.session_state.light_states.get(nid, "red")
            dot_color = PHASE_COLORS[phase]
            dot_label = PHASE_LABELS[phase]
            auto_note = "Manuell/Auto-Zyklus"

        scatter_data.append(
            {
                "lat": float(row["lat"]),
                "lon": float(row["lon"]),
                "label": row["unique_label"],
                "phase_label": f"Phase: {dot_label}",
                "auto_note": auto_note,
                "tooltip": "",
                "color": dot_color,
                "radius": (
                    17 if SHOW_MANUAL_TRAFFIC_LIGHT_CONTROLS and nid == st.session_state.selected_node
                    else (14 if (is_auto_green or is_auto_red) else 10)
                ),
                "line_color": (
                    [0, 0, 0, 255]
                    if SHOW_MANUAL_TRAFFIC_LIGHT_CONTROLS and nid == st.session_state.selected_node
                    else [0, 0, 0, 0]
                ),
            }
        )

    layers = [
        pdk.Layer(
            "PathLayer",
            data=edge_paths_df[["path"]].to_dict("records"),
            get_path="path",
            get_color=[65, 65, 95, 70],
            get_width=2,
            width_scale=1,
            width_min_pixels=1,
        )
    ]

    if not path_data.empty:
        map_paths = path_data.copy()
        # Ensure the path data has the same keys expected by the tooltip HTML
        map_paths["label"] = ""
        map_paths["phase_label"] = ""
        map_paths["auto_note"] = ""
        if map_mode == "Laständerung durch Intervention":
            map_paths["map_color"] = map_paths["load_change_color"]
            map_paths["map_width"] = map_paths["load_change_width"]
        else:
            map_paths["map_color"] = map_paths["congestion_map_color"]
            map_paths["map_width"] = map_paths["congestion_map_width"]

        layers.append(
            pdk.Layer(
                "PathLayer",
                data=map_paths.to_dict("records"),
                get_path="path",
                get_color="map_color",
                get_width="map_width",
                width_scale=1,
                width_min_pixels=2,
                pickable=True,
                auto_highlight=True,
            )
        )

    layers.append(
        pdk.Layer(
            "ScatterplotLayer",
            data=scatter_data,
            pickable=True,
            auto_highlight=True,
            get_position=["lon", "lat"],
            get_fill_color="color",
            get_radius="radius",
            radius_scale=3,
            radius_min_pixels=5,
            radius_max_pixels=26,
            stroked=True,
            get_line_color="line_color",
            line_width_min_pixels=2,
        )
    )

    deck = pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(
            longitude=centre_lon,
            latitude=centre_lat,
            zoom=14,
            pitch=0,
            bearing=0,
            min_zoom=12,
            max_zoom=18,
        ),
        map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
        tooltip={
            "html": (
                "<div style='font-family:system-ui,sans-serif;font-size:13px;line-height:1.55;'>"
                "{tooltip}"
                "<b>{label}</b><br/>{phase_label}<br/>{auto_note}"
                "</div>"
            ),
            "style": {
                "backgroundColor": "rgba(15,15,28,0.94)",
                "color": "#ffffff",
                "padding": "8px 14px",
                "borderRadius": "8px",
                "boxShadow": "0 2px 12px rgba(0,0,0,.4)",
            },
        },
    )
    st.pydeck_chart(deck, width="stretch", key="forecast_signal_map")


# ─────────────────────────────────────────────────────────────────────────────
# metrics
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("### Wirkung der Ampel-Intervention")

if result.empty:
    st.info("Lade eine Forecast-DB und wähle einen Prognosezeitpunkt, um die Intervention zu berechnen.")
else:
    try:
        before_congested = int((result["ForecastCongestionPercent"] >= threshold_percent).sum())
        after_congested = int((result["AdjustedCongestionPercent"] >= threshold_percent).sum())
        
        prevented = int(result["CongestionPrevented"].sum())
        green_count = int(result["GreenPriority"].sum())
        burden_count = int(result["RedBurden"].sum())
        total_relief = float(result["RelievedLoad"].sum())
        total_burden = float(result["AddedNeighborBurden"].sum())

        m1, m2, m4, m5, m3 = st.columns(5)
        m1.metric("Vorher über Stauchwelle", before_congested)
        m2.metric("Nachher über Stauschwelle", after_congested, delta=after_congested - before_congested)
        m4.metric("Grünpriorisierte Segmente", green_count)
        m5.metric("Rot-belastete Nachbarn", burden_count)
        m3.markdown(
            f"""
            <div style="border: 1px solid rgba(128, 128, 128, 0.4); padding: 10px; border-radius: 8px; font-size: 0.8rem; color: #888; width: 100%; margin-top: 10px; line-height: 1.4;">
            <b>Entlastet:</b> {total_relief:.1f} Fahrzeuge<br>
            <b>Nachbarbelastung:</b> {total_burden:.1f} Fahrzeuge<br><br>
            Synthetische Werte aus Forecast, keine reale Ampelberechnung.
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("<br>", unsafe_allow_html=True)

        with st.expander("Stärkste Grünprioritäten"):
            green_table = result[result["GreenPriority"].eq(1)].copy()
            green_table = green_table.sort_values("RelievedLoad", ascending=False).head(12)
            if green_table.empty:
                st.write("Keine Grünpriorität notwendig.")
            else:
                green_table = green_table[[
                        "Street", "ForecastCongestionPercent", "AdjustedCongestionPercent",
                        "RelievedLoad", "CongestionPrevented", "Segment",
                    ]]
                
                green_table = green_table.sort_values(by="Street", ascending=True)
                stau_bool_mapping = {
                    0: "Nein | kein Stau vorhergesagt",
                    1: "Ja",
                }
                green_table["CongestionPrevented"] = green_table["CongestionPrevented"].map(stau_bool_mapping)
                green_table["RelievedLoad"] = green_table["RelievedLoad"].round()
                green_table["ForecastCongestionPercent"] = green_table["ForecastCongestionPercent"].round(2)
                green_table["AdjustedCongestionPercent"] = green_table["AdjustedCongestionPercent"].round(2)
                
                # rename colums
                st.dataframe(
                    green_table.rename(columns={
                        "Street": "Straße",
                        "ForecastCongestionPercent": "Stau vor Eingriff (%)",
                        "AdjustedCongestionPercent": "Stau nach Eingriff (%)",
                        "RelievedLoad": "Anzahl entlastete Fz.",
                        "CongestionPrevented": "Stau verhindert?",
                    }),
                    hide_index=True,
                    width="stretch"
                )

        with st.expander("Nachbarbelastung durch mehr Rotzeit"):
            red_table = result[result["RedBurden"].eq(1)].copy()
            red_table = red_table.sort_values("AddedNeighborBurden", ascending=False).head(12)
            if red_table.empty:
                st.write("Keine Nachbarbelastung erzeugt.")
            else:
                red_table = red_table[[
                        "Street", "ForecastCongestionPercent", "AdjustedCongestionPercent",
                        "AddedNeighborBurden", "InterventionReason", "Segment",
                    ]]
                
                red_table = red_table.sort_values(by="Street", ascending=True)
                red_table["AddedNeighborBurden"] = red_table["AddedNeighborBurden"].round()
                red_table["ForecastCongestionPercent"] = red_table["ForecastCongestionPercent"].round(2)
                red_table["AdjustedCongestionPercent"] = red_table["AdjustedCongestionPercent"].round(2)

                # rename colums
                st.dataframe(
                    red_table.rename(columns={
                        "Street": "Straße",
                        "ForecastCongestionPercent": "Stau vor Eingriff (%)",
                        "AdjustedCongestionPercent": "Stau nach Eingriff (%)",
                        "AddedNeighborBurden": "Belastung d. Nachbarstraßen in Fz.",
                        "InterventionReason": "Grund für d. Eingriff",
                    }),
                    hide_index=True,
                    width="stretch"
                )
    except KeyError:
        st.caption(
            f"No prediction"
        )
    if SHOW_DEBUG_TABLES_AND_EXPORT:
        with st.expander("Alle berechneten Segmente anzeigen"):
            st.dataframe(
                result[[
                    "Timestamp", "Street", "Segment", "ForecastLoad", "AdjustedLoad", "DeltaLoad",
                    "ForecastCongestionPercent", "AdjustedCongestionPercent", "InterventionRole",
                    "InterventionReason",
                ]].sort_values("AdjustedCongestionPercent", ascending=False),
                width='stretch',
                hide_index=True,
            )

        export_col1, export_col2 = st.columns([1, 3])
        with export_col1:
            if st.button("Intervention in DB speichern", type="primary"):
                try:
                    export_adjusted_result(db_path, result)
                    st.success("Tabelle `traffic_signal_intervention` wurde gespeichert.")
                except Exception as exc:
                    st.error(f"Export fehlgeschlagen: {exc}")
        with export_col2:
            st.caption("Der Export überschreibt nur die Tabelle `traffic_signal_intervention`, nicht die ursprünglichen Forecast-Tabellen.")



# ─────────────────────────────────────────────────────────────────────────────
# AUTO-CYCLE
# ─────────────────────────────────────────────────────────────────────────────

if SHOW_MANUAL_TRAFFIC_LIGHT_CONTROLS and st.session_state.auto_cycle and not st.session_state.use_forecast_control:
    time.sleep(1.8)
    rng = np.random.default_rng()
    for nid in list(st.session_state.light_states.keys()):
        if rng.random() > 0.72:
            curr = st.session_state.light_states[nid]
            nxt = PHASES[(PHASES.index(curr) + 1) % len(PHASES)]
            st.session_state.light_states[nid] = nxt
    st.rerun()
