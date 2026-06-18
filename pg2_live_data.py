import streamlit as st
import osmnx as ox
import pandas as pd
import sqlite3
import pydeck as pdk
import time
from datetime import datetime
from streamlit_theme import st_theme

# ─────────────────────────────────────────────────────────────────────────────
# adjust to light & dark mode
# ─────────────────────────────────────────────────────────────────────────────
theme = st_theme()

bg_color = "#1a1a2e"
text_color = "#eee"
border = "none"

# colors in light mode
if theme and theme.get("base") == "light":
    bg_color = "#edf2fa"
    text_color = "#1a1a2e"
    border = "1px solid #c6d4ea"

# implement new style
st.markdown(
    """
    <style>
    .block-container, .stMainBlockContainer {
        max-width: 90% !important;
        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;
        padding-top: 1rem !important;
    }

    /* Legende in der oberen Leiste: light/dark-adaptiv */
    .pg2-legende {
        display: flex;
        flex-wrap: wrap;
        gap: 28px;
        align-items: center;
        height: 100%;
        padding: 8px 0;
        font-size: 0.84rem;
        color: var(--text-color);
    }

    /* Zeitstempel: light/dark-adaptiv */
    .ts-container { 
        text-align: right; 
        padding: 4px 0; 
        line-height: 1.45; 
    }
    
    .ts-sub { 
        font-size: 0.70rem; 
        color: var(--text-color);
        opacity: 0.7;
        letter-spacing: .06em; 
        text-transform: uppercase; 
    }
    
    .ts-main { 
        font-size: 1.18rem; 
        font-weight: 700; 
        color: var(--text-color);
        font-variant-numeric: tabular-nums; 
        letter-spacing: .04em; 
    }

    /* Pulsierender Punkt für Live-Indikator */
    @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.25} }
    .live-dot {
        display: inline-block;
        width: 9px; height: 9px;
        border-radius: 50%;
        background: #ff3333;
        animation: pulse 1.3s ease-in-out infinite;
        margin-right: 5px;
        vertical-align: middle;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <style>
    .strassentyp-legende {{
        padding: 10px 14px;
        border-radius: 10px;
        font-size: 0.80rem;
        line-height: 1.7;
        margin-top: 10px;
        background: {bg_color};
        color: {text_color};
        border: {border};
        transition: background 0.3s ease, color 0.3s ease;
    }}
    .strassentyp-legende table {{
        width: 100%;
        border-collapse: collapse;
        margin-top: 6px;
    }}
    .strassentyp-legende td {{
        padding: 2px 6px;
        vertical-align: top;
    }}
    .strassentyp-legende td:first-child {{
        font-weight: 600;
        white-space: nowrap;
        width: 38%;
    }}
    .methodik-box {{
        padding: 10px 14px;
        border-radius: 10px;
        font-size: 0.80rem;
        line-height: 1.65;
        margin-top: 10px;
        background: {bg_color};
        color: {text_color};
        border: {border};
        transition: background 0.3s ease, color 0.3s ease;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    "<h1 style='text-align: center;'>Live-Daten</h1>"
    "<br>", 
    unsafe_allow_html=True
)


# ─────────────────────────────────────────────────────────────────────────────
# load data  (gecacht --> only one DB / OSM query 
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data
def load_traffic_data() -> pd.DataFrame:
    conn = sqlite3.connect("traffic_data_darmstadt_mitte.db") # data from simul_data.py
    df = pd.read_sql("SELECT * FROM traffic_darmstadt", conn)
    conn.close()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    return df


@st.cache_resource
def load_graph(): # get street data
    return ox.graph_from_place(
        "Darmstadt-Mitte, Darmstadt, Germany", network_type="drive"
    )


@st.cache_data
def load_edges() -> pd.DataFrame:
    G = load_graph()
    _, edges = ox.graph_to_gdfs(G)
    return edges.reset_index()


df    = load_traffic_data()
edges = load_edges()


# definitions for all relevant highway types
HIGHWAY_DESCRIPTIONS = {
    "motorway":       "Autobahn - höchste Kapazität, kreuzungsfrei",
    "motorway_link":  "Autobahn-Auffahrt / -Abfahrt",
    "trunk":          "Schnellstraße - ähnlich Autobahn, wenige Kreuzungen",
    "trunk_link":     "Schnellstraßen-Auffahrt / -Abfahrt",
    "primary":        "Hauptstraße - verbindet größere Städte",
    "primary_link":   "Hauptstraßen-Abzweigung",
    "secondary":      "Bundesstraße - verbindet Städte & Gemeinden",
    "secondary_link": "Bundesstraßen-Abzweigung",
    "tertiary":       "Kreisstraße - verbindet kleinere Ortschaften",
    "tertiary_link":  "Kreisstraßen-Abzweigung",
    "residential":    "Wohnstraße - innerörtliche Nebenstraße",
    "service":        "Zufahrt - Parkplätze, Höfe, Einfahrten",
    "unclassified":   "Sonstige - kleine Nebenstraße",
    "living_street":  "Spielstraße - Schrittgeschwindigkeit, Fußgänger haben Vorrang",
    "pedestrian":     "Fußgängerzone - eingeschränkter Kfz-Verkehr",
    "track":          "Feldweg / Forststraße",
    "busway":         "Busspur",
    "road":           "Straße unbekannten Typs",
}

# OSMnx sometimes stores highway as a list - flatten to get all unique types
_hw_types: set[str] = set()
for val in edges["highway"].dropna():
    if isinstance(val, list):
        _hw_types.update(str(v) for v in val)
    else:
        _hw_types.add(str(val))

present_highway_types = sorted(_hw_types)

# centre the map
bounds     = edges.total_bounds          # (minx, miny, maxx, maxy)
centre_lon = (bounds[0] + bounds[2]) / 2
centre_lat = (bounds[1] + bounds[3]) / 2

# ─────────────────────────────────────────────────────────────────────────────
# set color scheme
# ─────────────────────────────────────────────────────────────────────────────
LEVEL_COLORS = {
    0: [  0,   0,   0,   0],   # transparent – keine Daten
    1: [  0, 204,  68, 235],   # grün        – freier Verkehr
    2: [255, 204,   0, 235],   # gelb        – dichter Verkehr
    3: [210,  20,  20, 235],   # rot         – stockender Verkehr
    4: [  0,   0,   0, 255],   # schwarz     – Stau
}

LEVEL_WIDTHS: dict[int, int] = {
    0: 4, 
    1: 5, 
    2: 6, 
    3: 8, 
    4: 10
}

FRAME_INTERVAL_S = 2   # seconds between frames in live mode

# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────

def _flatten(val) -> str: # OSMnx sometimes returns lists; turn them into a readable string.
    if isinstance(val, list):
        return ", ".join(str(v) for v in val) if val else "Unbekannt"
    return str(val) if pd.notna(val) else "Unbekannt"


def build_path_records(merged: pd.DataFrame) -> list[dict]: # convert a merged GeoDataFrame into a flat list of dicts
    records: list[dict] = []
    for _, row in merged.iterrows():
        geom  = row.geometry
        level = int(row["Stau_Level"])

        meta = {
            "name":    _flatten(row.get("name",    "Unbekannt")),
            "highway": _flatten(row.get("highway", "Unbekannt")),
            "level":   level,
            "cars": (
                str(int(row["Anzahl_Autos"]))
                if "Anzahl_Autos" in merged.columns
                and pd.notna(row.get("Anzahl_Autos"))
                else "–"
            ),
            "speed": (
                f"{row['Durchschnittsgeschwindigkeit_kmh']:.0f}"
                if "Durchschnittsgeschwindigkeit_kmh" in merged.columns
                and pd.notna(row.get("Durchschnittsgeschwindigkeit_kmh"))
                else "–"
            ),
            "color": LEVEL_COLORS[level],
            "width": LEVEL_WIDTHS[level],
        }
        lines = (
            [geom] if geom.geom_type == "LineString" else list(geom.geoms)
        )
        for line in lines:
            records.append({**meta, "path": [[c[0], c[1]] for c in line.coords]})

    return records


# ─────────────────────────────────────────────────────────────────────────────
# session state  (persists selected timestamp across reruns)
# ─────────────────────────────────────────────────────────────────────────────

if "live_mode" not in st.session_state:
    st.session_state.live_mode = False
if "frame_idx" not in st.session_state:
    st.session_state.frame_idx = 0

# ─────────────────────────────────────────────────────────────────────────────
# time window (last DB day  ×  current system time ± 2 h)
# Modulo keeps every slot on the same calendar date even near midnight.
# ─────────────────────────────────────────────────────────────────────────────

now = datetime.now()
last_day = df["Timestamp"].max()  # actual last entry in the DB

hours_window: list[pd.Timestamp] = [
    last_day - pd.Timedelta(hours=d)
    for d in range(4, -1, -1)   # last_ts-4h, -3h, -2h, -1h, 0h
]


ts:pd.Timestamp = hours_window[st.session_state.frame_idx % len(hours_window)]

# ─────────────────────────────────────────────────────────────────────────────
# top bar (above map)
# ─────────────────────────────────────────────────────────────────────────────

col_live, col_legend, col_time = st.columns([1, 6, 2])

# Live toggle button
with col_live:
    btn_label = "⏹ LIVE" if st.session_state.live_mode else "▶ LIVE"
    if st.button(
        btn_label,
        use_container_width=True,
        type="primary" if st.session_state.live_mode else "secondary",
        help="Live-Modus starten / stoppen",
    ):
        st.session_state.live_mode = not st.session_state.live_mode
        st.session_state.frame_idx = 0
        st.rerun()

# legend 
with col_legend:
    st.markdown(
        """
        <div class='pg2-legende'>
            <div><span style='color:#00cc44; font-size:1.3rem; vertical-align:middle;'>●</span> Freier Verkehr</div>
            <div><span style='color:#ffcc00; font-size:1.3rem; vertical-align:middle;'>●</span> Dichter Verkehr</div>
            <div><span style='color:#d21414; font-size:1.3rem; vertical-align:middle;'>●</span> Stockender Verkehr</div>
            <div><span style='color:#000000; font-size:1.3rem; vertical-align:middle; -webkit-text-stroke: 0.5px white;'>●</span> Stau</div>
            </div>
        """,
        unsafe_allow_html=True,
    )

# Date / time display
with col_time:
    if st.session_state.live_mode:
        sub = f"Zeitfenster {st.session_state.frame_idx + 1} / {len(hours_window)}"
    else:
        sub = f"Letzter Datenbankstand &nbsp;·&nbsp; {last_day}"

    st.markdown(
        f"""
        <div class='ts-container'>
            <span class='ts-sub'>{sub}</span><br>
            <span class='ts-main'>
                {ts.strftime("%d.%m.%Y")}
                &nbsp;&nbsp;
                {ts.strftime("%H:%M")}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# filter & merge --> only the data during the selected datetime
# ─────────────────────────────────────────────────────────────────────────────

filtered_df = df[df["Timestamp"].between(ts, ts + pd.Timedelta(minutes=59))]

merged = edges.merge(filtered_df, on=["u", "v", "key"], how="left")
merged["Stau_Level"] = merged["Stau_Level"].fillna(0).astype(int)
for col in ("name", "highway"):
    if col not in merged.columns:
        merged[col] = "Unbekannt"

path_data = build_path_records(merged)

# ─────────────────────────────────────────────────────────────────────────────
# pydeck map
# ─────────────────────────────────────────────────────────────────────────────

layer = pdk.Layer(
    "PathLayer",
    data=path_data,
    pickable=True,
    auto_highlight=True,
    get_path="path",
    get_color="color",
    get_width="width",
    width_min_pixels=2,
    width_scale=1,
    joint_rounded=True,
    cap_rounded=True,
)

view_state = pdk.ViewState(
    longitude=centre_lon,
    latitude=centre_lat,
    zoom=14,
    pitch=0,
    bearing=0,
    min_zoom=11,
    max_zoom=18,
)

# on hover
deck = pdk.Deck(
    layers=[layer],
    initial_view_state=view_state,
    map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
    tooltip={
        "html": (
            "<div style='font-family:system-ui,sans-serif; font-size:13px;"
            " line-height:1.6;'>"
            "<b style='font-size:14px;'>{name}</b><br/>"
            "Typ: {highway}<br/>"
            "Verkehrsstufe: <b>{level}</b><br/>"
            "Fahrzeuge: {cars}<br/>"
            "Ø Geschwindigkeit: {speed} km/h"
            "</div>"
        ),
        "style": {
            "backgroundColor": "rgba(15, 15, 28, 0.93)",
            "color": "#ffffff",
            "padding": "10px 14px",
            "borderRadius": "8px",
            "boxShadow": "0 2px 12px rgba(0,0,0,0.4)",
        },
    },
)

st.pydeck_chart(deck, width="stretch")

# ─────────────────────────────────────────────────────────────────────────────
# metrics
# ─────────────────────────────────────────────────────────────────────────────

if len(filtered_df) > 0:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Straßen mit Daten", len(filtered_df))

    if "Durchschnittsgeschwindigkeit_kmh" in filtered_df.columns:
        avg_v = filtered_df["Durchschnittsgeschwindigkeit_kmh"].mean()
        c2.metric("Ø Geschwindigkeit", f"{avg_v:.1f} km/h")

    if "Stau_Level" in filtered_df.columns:
            c3.metric("🔴 Stockender Verkehr",  int((filtered_df["Stau_Level"] == 3).sum()))
            c4.metric("⚫ Stau",     int((filtered_df["Stau_Level"] == 4).sum()))
else:
    st.info(f"Keine Verkehrsdaten für {ts.strftime('%d.%m.%Y %H:%M')} verfügbar.")

# ─────────────────────────────────────────────────────────────────────────────
# live auto-advance (sleep, increment frame, rerun whole script)
# last thing that runs so map renders before sleeping
# ─────────────────────────────────────────────────────────────────────────────

if st.session_state.live_mode:
    time.sleep(FRAME_INTERVAL_S)
    st.session_state.frame_idx = (st.session_state.frame_idx + 1) % len(hours_window)
    st.rerun()



###############################################################################################
'---'

legende, definitionen = st.columns(2)

# explain all street types
with legende:
    _rows = "".join(
        f"<tr><td>{ht}</td><td>{HIGHWAY_DESCRIPTIONS.get(ht, 'Keine Beschreibung verfügbar')}</td></tr>"
        for ht in present_highway_types
    )
    st.markdown(
        f"""
        <div class='strassentyp-legende'>
        <b style='display:block; margin-bottom:4px; font-size:1rem;'>Straßentypen im Kartenausschnitt</b>
        <table>{_rows}</table>
        </div>
        """,
        unsafe_allow_html=True,
    )

# explain values on hover
with definitionen:
    st.markdown(
    """
    <div class='methodik-box'>
    <b style='display:block; margin-bottom:4px; font-size:1rem;'>So werden die Werte berechnet</b>
    <b>Fahrzeuganzahl</b><br>
    Gibt an, wie viele Fahrzeuge sich im ausgewählten Zeitraum
    auf einem Straßenabschnitt befinden. Der Wert berücksichtigt
    Tageszeit, Straßentyp sowie den Einfluss umliegender
    Verkehrsschwerpunkte wie Innenstadt oder Einkaufsbereiche.<br><br>
    <b>Durchschnittsgeschwindigkeit</b><br>
    Die mittlere Fahrgeschwindigkeit der Fahrzeuge auf einem
    Abschnitt im gewählten Stundenzeitraum im Vergleich mit dem dort geltenden Tempolimit.
    Je stärker die Geschwindigkeit abweicht, desto höher
    ist das angezeigte Stau-Level.
    </div>
    """,
    unsafe_allow_html=True,
    )

