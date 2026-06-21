from __future__ import annotations
import math
import os
import re
import sqlite3
from dataclasses import dataclass
from datetime import timedelta
from typing import Dict, Iterable, List, Optional, Tuple
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SimulationConfig: 
    place: str = "Darmstadt-Mitte, Darmstadt, Germany"
    network_type: str = "drive"

    start: str = "2026-03-23 00:00"
    end: str = "2026-06-23 09:00"
    freq: str = "h"

    random_seed: int = 42

    # How strongly the previous hour influences the current hour
    # Higher values = smoother, more persistent traffic
    temporal_persistence: float = 0.68

    # Strength of hotspot demand in cars/hour
    hotspot_scale: float = 165.0

    # Strength of simple OD route pressure in cars/hour
    route_scale: float = 1.0

    # Strength of spillover from neighbouring congested segments
    spillover_strength: float = 0.18

    # Random noise applied to generated demand
    noise_std_fraction: float = 0.08

    # Random incident settings
    incident_probability_per_day: float = 0.85
    incident_duration_min_hours: int = 2
    incident_duration_max_hours: int = 5
    incident_capacity_factor_min: float = 0.45
    incident_capacity_factor_max: float = 0.75

    # BPR(Bureau of Public Roads function)-like speed model, higher beta creates sharper speed drops near capacity
    bpr_alpha: float = 0.55 # baseline sensitivity to growing traffic
    bpr_beta: float = 3.2

    output_db: str = "traffic_data_darmstadt_mitte.db"
    output_table: str = "traffic_darmstadt"
    write_chunk_size: int = 50_000


@dataclass(frozen=True)
class HotspotSpec:
    name: str
    lat: float
    lon: float
    kind: str
    strength: float
    decay_m: float


@dataclass(frozen=True)
class RouteSpec:
    name: str
    source_node: int
    target_node: int
    kind: str
    base_demand: float


# stable high-demand areas whose influence decays through the road graph
# these are defined manually. The segment closest to the specified coordinates will receive additional 
# demand based on the kind and strength, as well as connected segments with a decay.
DEFAULT_HOTSPOTS = [
    HotspotSpec(
        name="Luisenplatz_City_Center",
        lat=49.8728,
        lon=8.6512,
        kind="city_center",
        strength=1.00,
        decay_m=650.0,
    ),
    HotspotSpec(
        name="Schloss_Marktplatz",
        lat=49.8739,
        lon=8.6554,
        kind="shopping",
        strength=0.82,
        decay_m=560.0,
    ),
    HotspotSpec(
        name="Wilhelminenstrasse_Office_Shopping",
        lat=49.8698,
        lon=8.6504,
        kind="office_shopping",
        strength=0.72,
        decay_m=520.0,
    ),
    HotspotSpec(
        name="Heidelbergstraße",
        lat=49.866797,
        lon=8.646543,
        kind="officeentry_shopping",
        strength=0.4,
        decay_m=520.0,
    ),
    HotspotSpec(
        name="Wilhelminenplatz",
        lat=49.867621,
        lon=8.652479,
        kind="office_shopping",
        strength=0.2,
        decay_m=520.0,
    ),
    HotspotSpec(
        name="Zinn-Ankauf",
        lat=49.869311,  
        lon=8.659113,
        kind="office_shopping",
        strength=0.16,
        decay_m=520.0,
    ),
]

# ─────────────────────────────────────────────────────────────────────────────
# Basic helpers
# ─────────────────────────────────────────────────────────────────────────────

def _first_if_list(value, default=None):
    """Returns first value of list if input is a list. Otherwise returns the input value."""
    if isinstance(value, list):
        return value[0] if value else default
    return default if value is None else value


def _parse_numeric(value, default: float) -> float:
    """Parse values like 50, '50', '50 km/h', ['30', '50']."""
    value = _first_if_list(value, default)
    if value is None:
        return default
    if isinstance(value, (int, float)) and not pd.isna(value):
        return float(value)
    match = re.search(r"\d+(?:\.\d+)?", str(value)) #regex to search for numbers in a string like "[10]"
    return float(match.group(0)) if match else default


def get_time_category(hour: int, day_of_week: int) -> int:
    """
    1 = night
    2 = morning
    3 = midday
    4 = evening
    5 = morning rush hour
    6 = evening rush hour
    """

    if day_of_week >= 5:  # weekend
        if 0 <= hour < 6:
            return 1
        if 6 <= hour < 10:
            return 2
        if 10 <= hour < 15:
            return 3
        return 4

    # weekday
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


def gaussian_hour(hour_float: float, center: float, width: float) -> float:
    " Generates demand curves with normal distribution."
    return math.exp(-((hour_float - center) ** 2) / (2.0 * width**2))


def general_time_factor(timestamp: pd.Timestamp) -> float:
    """Smooth general daily demand curve."""
    hour = timestamp.hour + timestamp.minute / 60.0
    weekend = timestamp.dayofweek >= 5

    morning = gaussian_hour(hour, 8.0, 1.25)
    lunch = gaussian_hour(hour, 12.5, 1.8)
    evening = gaussian_hour(hour, 17.0, 1.55)
    night_low = gaussian_hour(hour, 3.0, 2.2)

    if weekend:
        # Weekends have weaker rush hours but stronger midday/evening activity
        factor = 0.48 + 0.20 * morning + 0.75 * lunch + 0.65 * evening - 0.12 * night_low
    else:
        factor = 0.55 + 1.15 * morning + 0.48 * lunch + 1.30 * evening - 0.18 * night_low

    return max(0.15, factor)


def hotspot_time_factor(kind: str, timestamp: pd.Timestamp) -> float:
    """Different hotspot types peak at different times."""
    hour = timestamp.hour + timestamp.minute / 60.0
    weekend = timestamp.dayofweek >= 5

    morning = gaussian_hour(hour, 8.0, 1.2)
    midday = gaussian_hour(hour, 12.8, 2.0)
    afternoon = gaussian_hour(hour, 16.0, 1.8)
    evening = gaussian_hour(hour, 18.2, 1.8)
    night = gaussian_hour(hour, 22.5, 2.2)

    if kind == "city_center":
        base = 0.50 + 0.45 * morning + 0.85 * midday + 0.75 * afternoon + 0.35 * evening
    elif kind == "shopping":
        base = 0.35 + 0.15 * morning + 1.00 * midday + 0.95 * afternoon + 0.20 * evening
    elif kind == "office_shopping":
        base = 0.35 + 0.95 * morning + 0.55 * midday + 0.90 * afternoon + 0.20 * evening
    elif kind == "entry":
        base = 0.45 + 0.90 * morning + 0.30 * midday + 1.05 * evening
    elif kind == "nightlife":
        base = 0.15 + 0.20 * evening + 1.15 * night
    else:
        base = 0.50 + 0.50 * midday

    if weekend:
        # Office traffic lower, shopping/leisure higher
        if kind in {"office_shopping", "entry"}:
            base *= 0.72
        if kind in {"city_center", "shopping", "nightlife"}:
            base *= 1.12

    return max(0.05, base)


def route_time_factor(kind: str, timestamp: pd.Timestamp) -> float:
    """Profiles for synthetic origin-destination corridor pressure."""
    hour = timestamp.hour + timestamp.minute / 60.0
    weekend = timestamp.dayofweek >= 5 #0-indexed. So the days go from 0 to 6

    morning = gaussian_hour(hour, 8.0, 1.1)
    midday = gaussian_hour(hour, 12.5, 2.3)
    evening = gaussian_hour(hour, 17.0, 1.35)
    late = gaussian_hour(hour, 20.5, 2.0)

    if kind == "inbound_morning":
        value = 0.20 + 1.35 * morning + 0.22 * midday
    elif kind == "outbound_evening":
        value = 0.18 + 0.25 * midday + 1.45 * evening
    elif kind == "shopping_inbound":
        value = 0.18 + 0.95 * midday + 0.65 * evening
    elif kind == "through":
        value = 0.25 + 0.55 * morning + 0.35 * midday + 0.62 * evening
    elif kind == "leisure_evening":
        value = 0.10 + 0.35 * evening + 0.85 * late
    else:
        value = 0.35 + 0.35 * midday

    if weekend:
        if kind in {"inbound_morning", "outbound_evening"}:
            value *= 0.55
        if kind in {"shopping_inbound", "leisure_evening"}:
            value *= 1.20

    return max(0.0, value)

# ─────────────────────────────────────────────────────────────────────────────
# Road parameters
# ─────────────────────────────────────────────────────────────────────────────

def road_parameters(edge_data: dict) -> Tuple[str, float, float, float, float]:
    """
    Return highway type, road multiplier, free speed, hourly capacity, lanes.
    """
    highway = str(_first_if_list(edge_data.get("highway"), "residential"))
    lanes = _parse_numeric(edge_data.get("lanes"), default=1.0)
    lanes = max(1.0, min(lanes, 4.0))

    # Prefer OSM maxspeed if available, otherwise use urban defaults
    fallback_speeds = {
        "motorway": 120.0,
        "motorway_link": 80.0,
        "trunk": 80.0,
        "trunk_link": 60.0,
        "primary": 50.0,
        "primary_link": 45.0,
        "secondary": 45.0,
        "secondary_link": 40.0,
        "tertiary": 40.0,
        "tertiary_link": 35.0,
        "unclassified": 35.0,
        "residential": 30.0,
        "living_street": 12.0,
        "service": 20.0,
    }
    free_speed = _parse_numeric(edge_data.get("maxspeed"), fallback_speeds.get(highway, 30.0))
    free_speed = max(8.0, min(free_speed, 130.0))

    # Approximate per-lane capacities in cars/hour
    per_lane_capacity = {
        "motorway": 1900.0,
        "motorway_link": 1200.0,
        "trunk": 1500.0,
        "trunk_link": 1000.0,
        "primary": 900.0,
        "primary_link": 650.0,
        "secondary": 700.0,
        "secondary_link": 550.0,
        "tertiary": 550.0,
        "tertiary_link": 420.0,
        "unclassified": 360.0,
        "residential": 260.0,
        "living_street": 120.0,
        "service": 110.0,
    }

    importance_multiplier = {
        "motorway": 2.60,
        "motorway_link": 1.90,
        "trunk": 2.20,
        "trunk_link": 1.65,
        "primary": 1.85,
        "primary_link": 1.45,
        "secondary": 1.45,
        "secondary_link": 1.25,
        "tertiary": 1.18,
        "tertiary_link": 1.05,
        "unclassified": 0.95,
        "residential": 0.78,
        "living_street": 0.40,
        "service": 0.36,
    }

    capacity = per_lane_capacity.get(highway, 240.0) * lanes
    multiplier = importance_multiplier.get(highway, 0.75)
    return highway, multiplier, free_speed, capacity, lanes

# ─────────────────────────────────────────────────────────────────────────────
# Network preparation
# ─────────────────────────────────────────────────────────────────────────────

def nearest_node_for_latlon(G: nx.MultiDiGraph, lat: float, lon: float) -> int:
    "Finds the nearest segment in the map based on longitude and latidute distance."
    return int(ox.distance.nearest_nodes(G, X=lon, Y=lat))

def choose_boundary_nodes(G: nx.MultiDiGraph) -> Dict[str, int]:
    """Pick rough north/south/east/west entry nodes from graph coordinates.
    Basically tries to determine which nodes are at the edge of the simulated 
    map and then sets them to be boundary nodes. One node is chosen for each cardinal direction, so 4 in total."""
    nodes = list(G.nodes(data=True))
    xs = np.array([float(d["x"]) for _, d in nodes])
    ys = np.array([float(d["y"]) for _, d in nodes])
    median_x = float(np.median(xs))
    median_y = float(np.median(ys))

    def best(candidates: Iterable[Tuple[int, dict]], score_fn) -> int:
        return int(min(candidates, key=lambda item: score_fn(item[1]))[0])

    north_cut = np.quantile(ys, 0.93)
    south_cut = np.quantile(ys, 0.07)
    east_cut = np.quantile(xs, 0.93)
    west_cut = np.quantile(xs, 0.07)

    north_candidates = [(n, d) for n, d in nodes if float(d["y"]) >= north_cut]
    south_candidates = [(n, d) for n, d in nodes if float(d["y"]) <= south_cut]
    east_candidates = [(n, d) for n, d in nodes if float(d["x"]) >= east_cut]
    west_candidates = [(n, d) for n, d in nodes if float(d["x"]) <= west_cut]
    #return the node most away from the centre of x or y from the candidate list. 
    # If the candidates were not separated, the code would not differentiate norht/south, east/west.
    return {
        "north": best(north_candidates, lambda d: abs(float(d["x"]) - median_x)),
        "south": best(south_candidates, lambda d: abs(float(d["x"]) - median_x)),
        "east": best(east_candidates, lambda d: abs(float(d["y"]) - median_y)),
        "west": best(west_candidates, lambda d: abs(float(d["y"]) - median_y)),
    }


def build_edge_table(G: nx.MultiDiGraph) -> pd.DataFrame:
    records = []
    
    for idx, (u, v, key, data) in enumerate(G.edges(keys=True, data=True)):
        #gives parameters to given roadtypes
        highway, multiplier, free_speed, capacity, lanes = road_parameters(data)
        street_name = _first_if_list(data.get("name"), "Unknown")
        length_m = float(data.get("length", 50.0))

        # Length factor prevents very tiny connector edges from dominating too much, but avoids making long roads unrealistically huge
        length_factor = max(0.65, min(1.40, math.sqrt(max(length_m, 10.0) / 90.0)))

        records.append(
            {
                "edge_idx": idx,
                "Segment": f"{u}_{v}_{key}",
                "u": int(u),
                "v": int(v),
                "key": int(key),
                "Strassenname": str(street_name),
                "Highway": highway,
                "Length_Meter": round(length_m, 2),
                "Lanes": lanes,
                "RoadImportance": multiplier * length_factor,
                "BaseSpeed_kmh": free_speed,
                "FlowCapacity": capacity,
            }
        )
    return pd.DataFrame(records)

def edge_index_lookup(G: nx.MultiDiGraph) -> Dict[Tuple[int, int, int], int]:
    "Gives every edge a number, referred to as idx."
    lookup = {}
    for idx, (u, v, key) in enumerate(G.edges(keys=True)):
        lookup[(int(u), int(v), int(key))] = idx
    return lookup

def shortest_edge_index_for_uv(
    G: nx.MultiDiGraph,
    lookup: Dict[Tuple[int, int, int], int],
    u: int,
    v: int,
) -> Optional[int]:
    """Finds the key of the edge with the shortest amount of length between the nodes u and v."""
    edge_bundle = G.get_edge_data(u, v)
    if not edge_bundle:
        return None
    best_key = min(edge_bundle.keys(), key=lambda k: float(edge_bundle[k].get("length", 1.0)))
    return lookup.get((int(u), int(v), int(best_key)))

def directed_edge_indices_for_node_path(
    G: nx.MultiDiGraph,
    lookup: Dict[Tuple[int, int, int], int],
    node_path: List[int],
) -> List[int]:
    """Converts a node path, as in a path from node a to node d via nodes b and c, into an edge path, 
    thus describing the same path via the edges between those nodes."""
    edge_indices = []
    for a, b in zip(node_path[:-1], node_path[1:]):
        edge_idx = shortest_edge_index_for_uv(G, lookup, int(a), int(b))
        if edge_idx is not None:
            edge_indices.append(edge_idx)
    return edge_indices

def build_edge_neighbours(G: nx.MultiDiGraph, edge_df: pd.DataFrame) -> List[List[int]]:
    outgoing_by_node: Dict[int, List[int]] = {}
    incoming_by_node: Dict[int, List[int]] = {}

    for row in edge_df.itertuples(index=False):
        outgoing_by_node.setdefault(int(row.u), []).append(int(row.edge_idx))
        incoming_by_node.setdefault(int(row.v), []).append(int(row.edge_idx))

    neighbours: List[List[int]] = []
    for row in edge_df.itertuples(index=False):
        edge_idx = int(row.edge_idx)
        # Neighbours that can feed into this edge or receive traffic from it.
        raw = incoming_by_node.get(int(row.u), []) + outgoing_by_node.get(int(row.v), [])
        cleaned = sorted({idx for idx in raw if idx != edge_idx})
        neighbours.append(cleaned)
    return neighbours

# ─────────────────────────────────────────────────────────────────────────────
# Hotspot and route pressure
# ─────────────────────────────────────────────────────────────────────────────

def build_hotspot_nodes(G: nx.MultiDiGraph, hotspot_specs: List[HotspotSpec]) -> List[Tuple[HotspotSpec, int]]:
    """Assigns hotspots to the nearest node. Hotspots and nodes carry with them coordinates."""
    result = []
    for spec in hotspot_specs:
        node = nearest_node_for_latlon(G, spec.lat, spec.lon)
        result.append((spec, node))
    return result


def build_hotspot_weight_matrix(
    G: nx.MultiDiGraph,
    edge_df: pd.DataFrame,
    hotspot_nodes: List[Tuple[HotspotSpec, int]],
) -> np.ndarray:
    """
    Matrix shape: [n_edges, n_hotspots].

    Influence decays by shortest path distance over an undirected version of the
    road network. This is better than a circular radius because disconnected but
    geographically close roads do not automatically receive equal influence.
    """
    Gu = G.to_undirected()
    n_edges = len(edge_df)
    weights = np.zeros((n_edges, len(hotspot_nodes)), dtype=float)

    for j, (spec, node) in enumerate(hotspot_nodes):
        cutoff = spec.decay_m * 5.0
        lengths = nx.single_source_dijkstra_path_length(Gu, node, cutoff=cutoff, weight="length")

        for row in edge_df.itertuples(index=False):
            du = lengths.get(int(row.u), float("inf"))
            dv = lengths.get(int(row.v), float("inf"))
            dist = min(du, dv)
            if math.isfinite(dist):
                weights[int(row.edge_idx), j] = spec.strength * math.exp(-dist / spec.decay_m)

    return weights


def build_route_specs(
    G: nx.MultiDiGraph,
    hotspot_nodes: List[Tuple[HotspotSpec, int]],
) -> List[RouteSpec]:
    boundary = choose_boundary_nodes(G)
    hotspot_by_kind = {spec.kind: node for spec, node in hotspot_nodes}
    city_node = hotspot_by_kind.get("city_center", hotspot_nodes[0][1])
    shopping_node = hotspot_by_kind.get("shopping", city_node)
    office_node = hotspot_by_kind.get("office_shopping", city_node)

    # These are deliberately simple. They create repeatable route corridors that a future redirection algorithm can try to avoid or rebalance
    return [
        RouteSpec("north_to_center_morning", boundary["north"], city_node, "inbound_morning", 115.0),
        RouteSpec("south_to_center_morning", boundary["south"], city_node, "inbound_morning", 105.0),
        RouteSpec("west_to_office_morning", boundary["west"], office_node, "inbound_morning", 95.0),
        RouteSpec("east_to_center_morning", boundary["east"], city_node, "inbound_morning", 85.0),
        RouteSpec("center_to_north_evening", city_node, boundary["north"], "outbound_evening", 110.0),
        RouteSpec("center_to_south_evening", city_node, boundary["south"], "outbound_evening", 105.0),
        RouteSpec("office_to_west_evening", office_node, boundary["west"], "outbound_evening", 95.0),
        RouteSpec("center_to_east_evening", city_node, boundary["east"], "outbound_evening", 85.0),
        RouteSpec("west_to_shopping", boundary["west"], shopping_node, "shopping_inbound", 75.0),
        RouteSpec("east_to_shopping", boundary["east"], shopping_node, "shopping_inbound", 70.0),
        RouteSpec("north_south_through", boundary["north"], boundary["south"], "through", 60.0),
        RouteSpec("west_east_through", boundary["west"], boundary["east"], "through", 55.0),
        RouteSpec("center_to_west_leisure", city_node, boundary["west"], "leisure_evening", 48.0),
    ]


def build_route_weight_matrix(
    G: nx.MultiDiGraph,
    edge_df: pd.DataFrame,
    route_specs: List[RouteSpec],
) -> np.ndarray:
    """Matrix shape [n_edges, n_routes], with 1.0 where a route uses an edge."""
    lookup = edge_index_lookup(G)
    n_edges = len(edge_df)
    weights = np.zeros((n_edges, len(route_specs)), dtype=float)

    for j, route in enumerate(route_specs):
        try:
            node_path = nx.shortest_path(G, route.source_node, route.target_node, weight="length")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue

        edge_path = directed_edge_indices_for_node_path(G, lookup, node_path)
        for edge_idx in edge_path:
            weights[edge_idx, j] = 1.0

        # Small route-radiation to immediate neighbours makes main corridors influence adjacent segments without pretending every adjacent road is equally part of the route
        used = set(edge_path)
        if used:
            neighbours = build_edge_neighbours(G, edge_df)
            for edge_idx in used:
                for nb in neighbours[edge_idx]:
                    if nb not in used:
                        weights[nb, j] = max(weights[nb, j], 0.25)

    return weights

# ─────────────────────────────────────────────────────────────────────────────
# Incidents and simulation
# ─────────────────────────────────────────────────────────────────────────────

def generate_incident_capacity_factors(
    timestamps: pd.DatetimeIndex,
    edge_df: pd.DataFrame,
    cfg: SimulationConfig,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return capacity factors and event flags.
    These are meant to simulate things like accidents or roadworks etc.
    factor = 1.0 means no incident.
    factor < 1.0 means temporary capacity reduction.
    """
    n_t = len(timestamps)
    n_e = len(edge_df)
    factors = np.ones((n_t, n_e), dtype=np.float32)
    flags = np.zeros((n_t, n_e), dtype=np.int8)

    # Incidents are more likely on important roads
    importance = edge_df["RoadImportance"].to_numpy(dtype=float)
    prob = importance / importance.sum()

    ts_series = pd.Series(np.arange(n_t), index=timestamps)
    unique_days = pd.Index(timestamps.normalize().unique())

    for day in unique_days:
        if rng.random() > cfg.incident_probability_per_day:
            continue

        # Usually one incident, sometimes two; that does not mean that there is always an incident, but rather that IF a day has one incident, it may get a second one.
        num_incidents = 1 + int(rng.random() < 0.18)
        day_indices = ts_series.loc[(ts_series.index >= day) & (ts_series.index < day + timedelta(days=1))].to_numpy()
        if len(day_indices) == 0:
            continue

        for _ in range(num_incidents): #apply incident to edge and give it a duration.
            edge_idx = int(rng.choice(edge_df["edge_idx"].to_numpy(), p=prob))
            start_idx = int(rng.choice(day_indices))
            duration = int(rng.integers(cfg.incident_duration_min_hours, cfg.incident_duration_max_hours + 1))
            end_idx = min(n_t, start_idx + duration)
            factor = float(rng.uniform(cfg.incident_capacity_factor_min, cfg.incident_capacity_factor_max))
            factors[start_idx:end_idx, edge_idx] = np.minimum(factors[start_idx:end_idx, edge_idx], factor)
            flags[start_idx:end_idx, edge_idx] = 1

    return factors, flags


def stau_level_from_speed_ratio(speed_ratio: np.ndarray) -> np.ndarray:
    return np.select(
        [speed_ratio >= 0.75, speed_ratio >= 0.50, speed_ratio >= 0.25],
        [1, 2, 3],
        default=4,
    ).astype(int)


def simulate(cfg: SimulationConfig) -> pd.DataFrame:
    """Simulates traffic based on config and all previously given sub-functions"""
    rng = np.random.default_rng(cfg.random_seed)

    print("Lade Straßennetz")
    G = ox.graph_from_place(cfg.place, network_type=cfg.network_type)
    print(f"{len(G.edges(keys=True))} Straßenabschnitte gefunden")

    timestamps = pd.date_range(start=cfg.start, end=cfg.end, freq=cfg.freq)
    edge_df = build_edge_table(G)
    n_edges = len(edge_df)

    print("Bereite Hotspots, Routen und Nachbarschaften vor")
    hotspot_nodes = build_hotspot_nodes(G, DEFAULT_HOTSPOTS)
    hotspot_matrix = build_hotspot_weight_matrix(G, edge_df, hotspot_nodes)

    route_specs = build_route_specs(G, hotspot_nodes)
    route_matrix = build_route_weight_matrix(G, edge_df, route_specs)

    neighbours = build_edge_neighbours(G, edge_df)
    incident_capacity_factors, incident_flags = generate_incident_capacity_factors(timestamps, edge_df, cfg, rng)

    road_importance = edge_df["RoadImportance"].to_numpy(dtype=float)
    base_speed = edge_df["BaseSpeed_kmh"].to_numpy(dtype=float)
    base_capacity = edge_df["FlowCapacity"].to_numpy(dtype=float)

    previous_volume = np.maximum(5.0, 0.15 * base_capacity * road_importance / np.maximum(road_importance.mean(), 0.01))
    previous_capacity_ratio = previous_volume / base_capacity

    all_chunks: List[pd.DataFrame] = []

    print("Simuliere Zeitreihendaten")
    for t_idx, timestamp in enumerate(timestamps):
        hour = timestamp.hour
        day_of_week = timestamp.dayofweek
        time_category = get_time_category(hour, day_of_week)
        weekend = int(day_of_week >= 5)

        general_factor = general_time_factor(timestamp)

        # Baseline demand per road type/importance
        baseline = 38.0 * general_factor * road_importance

        # Hotspot demand with network-distance decay
        hotspot_factors = np.array(
            [hotspot_time_factor(spec.kind, timestamp) for spec, _ in hotspot_nodes],
            dtype=float,
        )
        hotspot_pressure = cfg.hotspot_scale * (hotspot_matrix @ hotspot_factors)

        # Route corridor demand
        route_factors = np.array(
            [route_time_factor(route.kind, timestamp) * route.base_demand for route in route_specs],
            dtype=float,
        )
        route_pressure = cfg.route_scale * (route_matrix @ route_factors)

        # Congestion spillover from neighbours in the previous hour
        spillover = np.zeros(n_edges, dtype=float)
        overloaded_previous = np.maximum(0.0, previous_capacity_ratio - 0.72)
        for edge_idx, nb_list in enumerate(neighbours):
            if nb_list:
                spillover[edge_idx] = (
                    cfg.spillover_strength
                    * base_capacity[edge_idx]
                    * float(np.mean(overloaded_previous[nb_list]))
                )

        raw_target = baseline + hotspot_pressure + route_pressure + spillover

        # Multiplicative noise grows with the target, but remains controlled
        noise = rng.normal(loc=0.0, scale=cfg.noise_std_fraction, size=n_edges)
        target_volume = raw_target * np.maximum(0.0, 1.0 + noise)

        # Temporal persistence: this is the main reason the generated data forms smooth time series instead of independent random hourly samples
        volume = cfg.temporal_persistence * previous_volume + (1.0 - cfg.temporal_persistence) * target_volume
        volume = np.maximum(0.0, volume)

        capacity_factor = incident_capacity_factors[t_idx].astype(float)
        effective_capacity = np.maximum(10.0, base_capacity * capacity_factor)
        capacity_ratio = volume / effective_capacity

        # BPR-like speed decline
        speed = base_speed / (1.0 + cfg.bpr_alpha * np.power(capacity_ratio, cfg.bpr_beta))
        speed += rng.normal(loc=0.0, scale=2.0, size=n_edges)
        speed = np.clip(speed, 3.0, base_speed * 1.10)
        speed = np.round(speed, 1)

        speed_ratio = speed / base_speed
        stau_level = stau_level_from_speed_ratio(speed_ratio)

        chunk = edge_df.copy()
        chunk["Timestamp"] = timestamp
        chunk["Date"] = timestamp.date().isoformat()
        chunk["Time"] = timestamp.time().strftime("%H:%M:%S")
        chunk["Minute"] = int((timestamp - timestamps[0]).total_seconds() // 60)
        chunk["Wochentag"] = day_of_week
        chunk["Day"] = day_of_week
        chunk["Hour"] = hour
        chunk["Tageszeit_Kategorie"] = time_category
        chunk["IsNonWorkingDay"] = weekend
        chunk["RushHourActive"] = int(time_category in {5, 6})

        chunk["Anzahl_Autos"] = np.rint(volume).astype(int)
        chunk["Load"] = np.rint(volume).astype(int)
        chunk["Durchschnittsgeschwindigkeit_kmh"] = speed
        chunk["SpeedKmh"] = speed
        chunk["Stau_Level"] = stau_level
        chunk["Congestion"] = (stau_level >= 3).astype(int)
        chunk["CongestionPercent"] = np.round(np.clip(capacity_ratio, 0.0, 2.0) * 100.0, 1)
        #to be changed
        chunk["CapacityRatio"] = np.round(capacity_ratio, 4)

        chunk["HotspotPressure"] = np.round(hotspot_pressure, 2)
        chunk["RoutePressure"] = np.round(route_pressure, 2)
        chunk["SpilloverPressure"] = np.round(spillover, 2)
        chunk["IncidentActive"] = incident_flags[t_idx].astype(int)
        chunk["IncidentCapacityFactor"] = np.round(capacity_factor, 3)
        chunk["EffectiveCapacity"] = np.round(effective_capacity, 1)

        # Retain original u/v/key naming and add more explicit aliases
        chunk["FromNode"] = chunk["u"]
        chunk["ToNode"] = chunk["v"]

        all_chunks.append(chunk)

        previous_volume = volume
        previous_capacity_ratio = capacity_ratio

    df = pd.concat(all_chunks, ignore_index=True)

    # Prefer stable column order for downstream code and easier inspection
    preferred_cols = [
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
    ]
    remaining_cols = [c for c in df.columns if c not in preferred_cols]
    return df[preferred_cols + remaining_cols]


def export_to_sqlite(df: pd.DataFrame, cfg: SimulationConfig) -> None:
    print("Exportiere in SQLite-Datenbank")
    output_dir = os.path.dirname(cfg.output_db)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with sqlite3.connect(cfg.output_db) as conn:
        # Replace table once, then append chunks if needed
        first = True
        for start in range(0, len(df), cfg.write_chunk_size):
            end = start + cfg.write_chunk_size
            chunk = df.iloc[start:end]
            chunk.to_sql(
                cfg.output_table,
                conn,
                if_exists="replace" if first else "append",
                index=False,
            )
            first = False

        conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{cfg.output_table}_time ON {cfg.output_table}(Timestamp)")
        conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{cfg.output_table}_segment ON {cfg.output_table}(Segment)")
        conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{cfg.output_table}_datetime_segment ON {cfg.output_table}(Date, Time, Segment)")

    print(f"Datenbank erstellt: {cfg.output_db}")
    print(f"Tabelle: {cfg.output_table}")
    print(f"Zeilen: {len(df):,}")


def main() -> None:
    """loads simulate with the provided config, then writes the result in a database."""
    cfg = SimulationConfig()
    df = simulate(cfg)
    export_to_sqlite(df, cfg)

    print("Fertig")
    print(df.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
