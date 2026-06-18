import streamlit as st
import osmnx as ox
import sqlite3
import pandas as pd
import pydeck as pdk
from streamlit_theme import st_theme
from info_overlay import render_info_overlay

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
    f"""
    <style>
    .block-container, .stMainBlockContainer {{
        max-width: 90% !important;
        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;
        padding-top: 1rem !important;
    }}

    .verkehr-legende {{
        padding: 10px 14px;
        border-radius: 10px;
        font-size: 0.83rem;
        line-height: 2;
        margin-top: 4px;
        background: {bg_color};
        color: {text_color};
        border: {border};
        transition: background 0.3s ease, color 0.3s ease;
    }}
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
    "<h1 style='text-align: center;'>Historische Verkehrsdaten</h1>"
    "<h2 style='text-align: center;'>der letzten 3 Monate</h2>"
    "<br><br>",
    unsafe_allow_html=True
)


# ─────────────────────────────────────────────────────────────────────────────
# load data  (gecacht --> only one DB / OSM query)
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

df = load_traffic_data()
edges = load_edges()

# OSMnx sometimes stores highway as a list - flatten to get all unique types
_hw_types: set[str] = set()
for val in edges["highway"].dropna():
    if isinstance(val, list):
        _hw_types.update(str(v) for v in val)
    else:
        _hw_types.add(str(val))

present_highway_types = sorted(_hw_types)

# get limits of DB
min_date = df["Timestamp"].min().date()
max_date = df["Timestamp"].max().date()

# centre the map
bounds = edges.total_bounds          # (minx, miny, maxx, maxy)
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

LEVEL_WIDTHS = {
    0: 4, 
    1: 5, 
    2: 6, 
    3: 8, 
    4: 10
}

# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────

def _flatten(val) -> str: # OSMnx sometimes returns lists; turn them into a readable string.
    if isinstance(val, list):
        return ", ".join(str(v) for v in val) if val else "Unbekannt"
    return str(val) if pd.notna(val) else "Unbekannt"


def build_path_records(merged: pd.DataFrame) -> list[dict]: # convert a merged GeoDataFrame into a flat list of dicts
    records = []
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
                else "-"
            ),
            "color": LEVEL_COLORS[level],
            "width": LEVEL_WIDTHS[level],
        }

        lines = (
            [geom] if geom.geom_type == "LineString"
            else list(geom.geoms)
        )
        for line in lines:
            records.append({**meta, "path": [[c[0], c[1]] for c in line.coords]})

    return records


# ─────────────────────────────────────────────────────────────────────────────
# session state  (persists selected timestamp across reruns)
# ─────────────────────────────────────────────────────────────────────────────
_min_dt = pd.Timestamp(min_date)
_max_dt = pd.Timestamp(max_date) + pd.Timedelta(hours=23, minutes=59)

if "pg1_selected_dt" not in st.session_state:
    st.session_state.pg1_selected_dt = _min_dt + pd.Timedelta(hours=8)
else:
    _cur = pd.Timestamp(st.session_state.pg1_selected_dt)
    if _cur < _min_dt:
        st.session_state.pg1_selected_dt = _min_dt + pd.Timedelta(hours=8)
    elif _cur > _max_dt:
        st.session_state.pg1_selected_dt = _max_dt

# ─────────────────────────────────────────────────────────────────────────────
# layout
# ─────────────────────────────────────────────────────────────────────────────
col_left, col_right = st.columns([2, 1])

# ─────────────────────────────────────────────────────────────────────────────
# col_right (input)
# ─────────────────────────────────────────────────────────────────────────────
with col_right:
    ctrl_prev, ctrl_dt, ctrl_next = st.columns([1, 6, 1])

    # 'previous' button
    with ctrl_prev:
        st.write("")   # align button vertically with the input
        if st.button("◀", width="stretch", help="Vorherige Stunde"):
            candidate = pd.Timestamp(st.session_state.pg1_selected_dt) - pd.Timedelta(hours=1)
            if candidate.date() >= min_date:
                st.session_state.pg1_selected_dt = candidate
                st.session_state.pg1_dt_widget = candidate  # keep widget in sync
    
    # 'next' button
    with ctrl_next:
        st.write("")
        if st.button("▶", width="stretch", help="Nächste Stunde"):
            candidate = pd.Timestamp(st.session_state.pg1_selected_dt) + pd.Timedelta(hours=1)
            if candidate.date() <= max_date:
                st.session_state.pg1_selected_dt = candidate
                st.session_state.pg1_dt_widget = candidate  # keep widget in sync

    # datetime input
    with ctrl_dt:
        selected = st.datetime_input(
            "Datum & Uhrzeit",
            value=st.session_state.pg1_selected_dt,
            min_value=min_date,
            max_value=max_date,
            step=pd.Timedelta(minutes=60),
            key="pg1_dt_widget",
        )
        # write the widget's value back so the map always reflects it
        st.session_state.pg1_selected_dt = pd.Timestamp(selected)
    
    # legend of traffic levels
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
    """
    <div class='verkehr-legende'>
    <b style='display:block; margin-bottom:4px;'>Verkehrsstufe</b>
    <div style='display:flex; flex-wrap:wrap; gap:15px;'>
        <div><span style='color:#00cc44; font-size:1.3rem; vertical-align:middle;'>●</span> Freier Verkehr</div>
        <div><span style='color:#ffcc00; font-size:1.3rem; vertical-align:middle;'>●</span> Dichter Verkehr</div>
        <div><span style='color:#d21414; font-size:1.3rem; vertical-align:middle;'>●</span> Stockender Verkehr</div>
        <div><span style='color:#000000; font-size:1.3rem; vertical-align:middle; -webkit-text-stroke: 0.5px white;'>●</span> Stau</div>
    </div>
    </div>
    """,
    unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# filter & merge --> only the data during the selected datetime
# ─────────────────────────────────────────────────────────────────────────────

ts = st.session_state.pg1_selected_dt

filtered_df = df[df["Timestamp"].between(ts, ts + pd.Timedelta(minutes=59))]

merged = edges.merge(filtered_df, on=["u", "v", "key"], how="left")
merged["Stau_Level"] = merged["Stau_Level"].fillna(0).astype(int)
for col in ("name", "highway"):
    if col not in merged.columns:
        merged[col] = "Unbekannt"

path_data = build_path_records(merged)

# ─────────────────────────────────────────────────────────────────────────────
# col_right (dashboard/metrics)
# ─────────────────────────────────────────────────────────────────────────────
 
with col_right:
    st.markdown("<br>", unsafe_allow_html=True)
    if len(filtered_df) > 0:
        c1, c2 = st.columns(2)
        c1.metric("Straßen mit Daten", len(filtered_df))

        # average speed of all streets during the selected datetime
        if "Durchschnittsgeschwindigkeit_kmh" in filtered_df.columns:
            avg_v = filtered_df["Durchschnittsgeschwindigkeit_kmh"].mean()
            c2.metric("Ø Geschwindigkeit", f"{avg_v:.1f} km/h")

        # absolute number of streets with red or black traffic level
        if "Stau_Level" in filtered_df.columns:
            c1.metric("🔴 Stockender Verkehr", int((filtered_df["Stau_Level"] == 3).sum()))
            c2.metric("⚫ Stau", int((filtered_df["Stau_Level"] == 4).sum()))
    else: # just in case
        st.info(f"Keine Verkehrsdaten für {ts.strftime('%d.%m.%Y %H:%M')} verfügbar.")

# ─────────────────────────────────────────────────────────────────────────────
# col_left (pydeck map)
# ─────────────────────────────────────────────────────────────────────────────
with col_left:
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
                "<div style='font-family:system-ui,sans-serif; font-size:13px; "
                "line-height:1.6;'>"
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

    st.pydeck_chart(deck, width='stretch')


# ─────────────────────────────────────────────────────────────────────────────
# filtered data
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)

# define relevant columns
active_columns = [
    "Timestamp", "Wochentag", "Tageszeit_Kategorie", 
    "Strassenname", "Length_Meter", "Lanes", 
    "Anzahl_Autos", "BaseSpeed_kmh", "Durchschnittsgeschwindigkeit_kmh"
]

# give str names to int values
weekday_mapping = {
    0: "Montag",
    1: "Dienstag",
    2: "Mittwoch",
    3: "Donnerstag",
    4: "Freitag",
    5: "Samstag",
    6: "Sonntag"
}

tageszeit_kat_mapping = {
    1: "Nacht",
    2: "Morgen",
    3: "Mittag",
    4: "Abend",
    5: "Rush Hour morgens",
    6: "Rush Hour abends"
}

display_df = filtered_df[active_columns].copy()
display_df["Wochentag"] = display_df["Wochentag"].map(weekday_mapping)
display_df["Tageszeit_Kategorie"] = display_df["Tageszeit_Kategorie"].map(tageszeit_kat_mapping)

# rename colums
st.dataframe(
    display_df.rename(columns={
        "Timestamp": "Datum",
        "Tageszeit_Kategorie": "Tageszeit-Kategorie",
        "Length_Meter": "Länge (m)",
        "Anzahl_Autos": "Anzahl Autos",
        "BaseSpeed_kmh": "Speed Limit (km/h)",
        "Lanes": "Anzahl Spuren",
        "Durchschnittsgeschwindigkeit_kmh": "Ø Geschwindigkeit (km/h)"
    }),
    hide_index=True
)


########################################################################

# explanations for street data from the map overlay
render_info_overlay(present_highway_types)