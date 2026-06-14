import streamlit as st
import pandas as pd
import pydeck as pdk
import osmnx as ox
from streamlit_theme import st_theme

st.markdown(
    "<h1 style='text-align: center;'>Verkehrsmanagementtool<br>der Stadt Darmstadt</h1>", 
    unsafe_allow_html=True
)


# ─────────────────────────────────────────────────────────────────────────────
# Navigation - adjust buttons in light & dark mode
# ─────────────────────────────────────────────────────────────────────────────
theme = st_theme()

bg_color = "#1a1a2e"
text_color = "#eee"
border = "none"

# button colors
btn_bg          = "rgba(255, 255, 255, 0.06)"
btn_border      = "rgba(255, 255, 255, 0.18)"
btn_hover_bg    = "rgba(255, 255, 255, 0.15)"
btn_hover_border= "rgba(255, 255, 255, 0.38)"

# buttons in light mode
if theme and theme.get("base") == "light":
    bg_color = "#edf2fa"
    text_color = "#1a1a2e"
    border = "1px solid #c6d4ea"

    btn_bg          = "#eef2f8"
    btn_border      = "#c6d2e4"
    btn_hover_bg    = "#dce7f5"
    btn_hover_border= "#8aadd0"

# implement button style
st.markdown(f"""
    <style>
    [data-testid="stPageLink"] a {{
        display: inline-block;
        width: 100%;
        padding: 0.4rem 0.8rem;
        border-radius: 0.5rem;
        text-align: center;
        text-decoration: none;
        color: inherit;
        font-weight: 500;
        transition: background-color 0.2s, border-color 0.2s;
        background-color: {btn_bg};
        border: 1px solid {btn_border};
    }}
    [data-testid="stPageLink"] a:hover {{
        background-color: {btn_hover_bg};
        border-color: {btn_hover_border};
    }}
    </style>
""", unsafe_allow_html=True)




# ─────────────────────────────────────────────────────────────────────────────
# Introduction
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("## Über dieses Tool")

st.markdown(
    """
    <div style="text-align: justify;">
    Das Verkehrsmanagementtool der Stadt Darmstadt ist ein datengetriebenes Dashboard 
    zur Analyse, Prognose und aktiven Steuerung des Straßenverkehrs in der Stadt 
    Darmstadt. Es basiert auf simulierten Verkehrsmessdaten und einem 
    Machine-Learning-Modell zur stündlichen Verkehrsprognose. Die interaktive Karte 
    zeigt alle Straßensegmente farblich nach Verkehrsstufe - von freiem Verkehr bis Stau.
    </div>
    """,
    unsafe_allow_html=True
)

"---"

# ─────────────────────────────────────────────────────────────────────────────
# page overview
# ─────────────────────────────────────────────────────────────────────────────

col_a, col_b = st.columns(2)

with col_a:
    col_a.page_link("pg1_hist_data.py", label="**Historische Verkehrsdaten**")
    st.markdown(
        "Analysiert die Verkehrslage der vergangenen drei Monate. Über die Zeitsteuerung "
        "lässt sich jede beliebige Stunde im Beobachtungszeitraum aufrufen. "
        "Die Karte zeigt alle Straßensegmente farblich nach Staulevel. "
        "Ein Klick auf ein Segment zeigt Fahrzeuganzahl und Durchschnittsgeschwindigkeit an."
    )
    st.markdown("<br>", unsafe_allow_html=True)
    
with col_b:
    col_b.page_link("pg2_live_data.py", label="**Live-Daten**")
    st.markdown(
        "Stellt die aktuellen Verkehrsdaten dar. Im Live-Modus durchläuft die Anzeige die verfügbaren Zeitschritte "
        "automatisch und die Karte aktualisiert sich fortlaufend."
    )
    

col_c, col_d = st.columns(2)

with col_c:
    col_c.page_link("pg3_forecast.py", label="**Verkehrsprognose**")
    st.markdown(
        "Zeigt die stündlich prognostizierten Verkehrsverhältnisse für die nächsten Stunden. "
        "Das Modell berechnet auf Basis historischer Muster für jedes Straßensegment "
        "Staulevel, Fahrzeuganzahl und Durchschnittsgeschwindigkeit."
    )

with col_d:
    col_d.page_link("pg4_traffic_lights_manual.py", label="**Ampelschaltung**")
    st.markdown(
        "Interaktive Steuerung der Ampeln im Stadtgebiet. Einzelne Kreuzungen "
        "lassen sich manuell schalten oder per Grüne-Welle-Funktion koordiniert freigeben. "
        "Der Auto-Zyklus simuliert den realistischen Phasenwechsel."
    )



"---"

# ─────────────────────────────────────────────────────────────────────────────
# traffic map (Darmstadt-Mitte)
# ─────────────────────────────────────────────────────────────────────────────

def traffic_map(location):
    # get street data
    G = ox.graph_from_place(location, network_type="drive")
    _, edges = ox.graph_to_gdfs(G)
    edges = edges.reset_index()

    def _flatten(val) -> str: # catch possible errors in street recognition
        if isinstance(val, list):
            return ", ".join(str(v) for v in val) if val else "Unbekannt"
        return str(val) if pd.notna(val) else "Unbekannt"

    # build path records for PyDeck PathLayer
    records = []
    for _, row in edges.iterrows():
        geom = row.geometry
        lines = [geom] if geom.geom_type == "LineString" else list(geom.geoms)
        for line in lines:
            records.append({
                "path": [[c[0], c[1]] for c in line.coords],
                "name": _flatten(row.get("name", "Unbekannt")),
                "highway": _flatten(row.get("highway", "Unbekannt")),
            })

    bounds = edges.total_bounds
    centre_lon = (bounds[0] + bounds[2]) / 2
    centre_lat = (bounds[1] + bounds[3]) / 2

    layer = pdk.Layer(
        "PathLayer",
        data=records,
        pickable=True,
        auto_highlight=True,
        get_path="path",
        get_color=[30, 120, 220, 200],
        get_width=6,
        width_min_pixels=1,
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

    # pop-up window on hover
    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
        tooltip={
            "html": (
                "<div style='font-family:system-ui,sans-serif; font-size:13px; line-height:1.6;'>"
                "<b style='font-size:14px;'>{name}</b><br/>"
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

st.markdown("## Straßenkarte")
with st.spinner("Karte wird geladen …"):
    traffic_map(location="Darmstadt-Mitte, Darmstadt, Germany")
