import streamlit as st
import osmnx as ox
import pandas as pd
import sqlite3
import pydeck as pdk
from streamlit_theme import st_theme
import locale

# Set the locale to German (for date string)
# Linux/macOS: 'de_DE.UTF-8'
# Windows: 'de_DE' or 'German'
try:
    locale.setlocale(locale.LC_TIME, 'de_DE.UTF-8')
except locale.Error:
    locale.setlocale(locale.LC_TIME, 'de_DE')

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
    /* ADD THESE TWO NEW CLASSES: */
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
    "<h1 style='text-align: center;'>Verkehrsprognose</h1>"
    "<h2 style='text-align: center;'>Prognose der nächsten 8? Stunden</h2>"
    "<br><br>",
    unsafe_allow_html=True
)


# ─────────────────────────────────────────────────────────────────────────────
# load data  (gecacht --> only one DB / OSM query)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data
def load_traffic_data() -> pd.DataFrame:
    conn = sqlite3.connect('traffic_forecasts.db') # forecast data
    df = pd.read_sql_query("SELECT * FROM verkehr_darmstadt_forecast", conn)
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

available_timestamps = sorted(pd.to_datetime(df["Timestamp"].unique()))
unique_dates = sorted(set(ts.date() for ts in available_timestamps))

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
                else "-"
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
if "pg2_selected_dt" not in st.session_state or st.session_state.pg2_selected_dt not in available_timestamps:
    st.session_state.pg2_selected_dt = available_timestamps[0]

def set_active_date(new_date): # triggered when a date button is clicked.
    first_of_day = next(t for t in available_timestamps if t.date() == new_date)
    st.session_state.pg2_selected_dt = first_of_day

def go_prev(): # triggered by the ◀ button.
    idx = available_timestamps.index(st.session_state.pg2_selected_dt)
    if idx > 0:
        st.session_state.pg2_selected_dt = available_timestamps[idx - 1]

def go_next(): # triggered by the ▶ button.
    idx = available_timestamps.index(st.session_state.pg2_selected_dt)
    if idx < len(available_timestamps) - 1:
        st.session_state.pg2_selected_dt = available_timestamps[idx + 1]

# ─────────────────────────────────────────────────────────────────────────────
# layout
# ─────────────────────────────────────────────────────────────────────────────
col_left, col_right = st.columns([2, 1])

# ─────────────────────────────────────────────────────────────────────────────
# col_right (input)
# ─────────────────────────────────────────────────────────────────────────────
with col_right:
    # Date of forecast
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
                is_active = (d == st.session_state.pg2_selected_dt.date())
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
    selected_date   = st.session_state.pg2_selected_dt.date()
    day_timestamps  = [t for t in available_timestamps if t.date() == selected_date]

    if st.session_state.pg2_selected_dt not in day_timestamps:
        st.session_state.pg2_selected_dt = day_timestamps[0]

    with ctrl_dt:
        st.selectbox(
            "Uhrzeit",
            key="pg2_selected_dt",   # session state IS the selected value
            options=day_timestamps,  # only this day's timestamps
            format_func=lambda ts: ts.strftime("%H:%M"),
        )
 
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
# filter & merge
# ─────────────────────────────────────────────────────────────────────────────
ts = st.session_state.pg2_selected_dt

filtered_df = df[df["Timestamp"] == ts]

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
    else:
        st.info(f"Keine Prognosedaten für {ts.strftime('%d.%m.%Y %H:%M')} verfügbar.")

# ─────────────────────────────────────────────────────────────────────────────
# col_left (pydeck map)
# ─────────────────────────────────────────────────────────────────────────────
with col_left:
    overlay_text = st.session_state.pg2_selected_dt.strftime("%a, %d.%m. | %H:%M Uhr")
    st.markdown(
        f"""
        <div style='position:absolute; top:25px; left:10px; z-index:99; 
                    background:rgba(15, 15, 28, 0.93); padding:10px 16px; 
                    border-radius:8px; border:1px solid rgba(255,255,255,0.2); 
                    color:white; font-size:1.15rem; font-weight:bold; 
                    box-shadow: 0 4px 12px rgba(0,0,0,0.3); pointer-events:none;'>
            {overlay_text}
        </div>
        """,
        unsafe_allow_html=True
    )

    layer = pdk.Layer(
        "PathLayer",
        data=path_data,
        pickable=True,
        auto_highlight=True,
        get_path="path",
        get_color="color",
        get_width="width",
        width_min_pixels=2,      # always at least 2 px wide when zoomed out
        width_scale=1,           # width values are already in metres
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
