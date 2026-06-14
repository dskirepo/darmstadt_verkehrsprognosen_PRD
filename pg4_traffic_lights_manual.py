import streamlit as st
import osmnx as ox
import pandas as pd
import numpy as np
import pydeck as pdk
import time

# wide layout
st.markdown(
    """
    <style>
    .block-container, .stMainBlockContainer {
        max-width: 90% !important;
        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;
        padding-top: 1rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    "<h1 style='text-align: center;'>Ampelschaltungssoftware</h1>"
    "<p style='text-align: center;'>Über diese Ansicht können Sie die Ampeln sämtlicher Kreuzungen überwachen und bei Bedarf manuell eingreifen.</p>",
    unsafe_allow_html=True,
)

st.markdown("<br><br>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# load data  (gecacht --> only one DB / OSM query)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Straßennetz wird geladen …")
def load_all_data():
    G = ox.graph_from_place(
        "Darmstadt-Mitte, Darmstadt, Germany", network_type="drive"
    )
    _, edges_gdf = ox.graph_to_gdfs(G)
    edges_gdf = edges_gdf.reset_index()

    # intersections: nodes with ≥ 3 connections
    rows = []
    for node_id in G.nodes():
        if G.degree(node_id) < 3:
            continue
        nd = G.nodes[node_id]
        streets: set = set()
        for _, _, ed in G.edges(node_id, data=True):
            name = ed.get("name", "")
            if isinstance(name, list):
                streets.update(n for n in name if n)
            elif name:
                streets.add(name)
        label = " / ".join(sorted(streets)[:2]) if streets else f"Knotenpunkt {node_id}"
        rows.append(
            {
                "node_id": int(node_id),
                "lat": float(nd["y"]),
                "lon": float(nd["x"]),
                "label": label,
            }
        )

    df_nodes = pd.DataFrame(rows)
    label_cnt = df_nodes["label"].value_counts()
    df_nodes["unique_label"] = df_nodes.apply(
        lambda r: f"{r['label']} (#{r['node_id']})"
        if label_cnt.get(r["label"], 0) > 1
        else r["label"],
        axis=1,
    )
    df_nodes = df_nodes.sort_values("unique_label").reset_index(drop=True)

    # edge paths for the background road layer
    edge_paths = []
    for _, row in edges_gdf.iterrows():
        geom = row.geometry
        if geom.geom_type == "LineString":
            lines = [geom]
        elif geom.geom_type == "MultiLineString":
            lines = list(geom.geoms)
        else:
            continue
        for line in lines:
            edge_paths.append(
                {"path": [[float(c[0]), float(c[1])] for c in line.coords]}
            )

    bounds = edges_gdf.total_bounds.tolist()
    return df_nodes, edge_paths, bounds


intersections, edge_paths, bounds = load_all_data()

if intersections.empty:
    st.error("Keine Kreuzungen gefunden")
    st.stop()

centre_lon = (bounds[0] + bounds[2]) / 2
centre_lat = (bounds[1] + bounds[3]) / 2


# ─────────────────────────────────────────────────────────────────────────────
# constants
# ─────────────────────────────────────────────────────────────────────────────
PHASES = ["red", "green", "yellow"]   # cycle order
PHASE_COLORS = {
    "red":    [210,  20,  20, 240],
    "yellow": [255, 200,   0, 240],
    "green":  [  0, 204,  68, 240],
}
PHASE_LABELS = {
    "red":    "Rot",
    "yellow": "Gelb",
    "green":  "Grün",
}

# ─────────────────────────────────────────────────────────────────────────────
# street names  –  built early so sync helpers work before the layout runs
# ─────────────────────────────────────────────────────────────────────────────
street_names = sorted(
    {
        part.strip()
        for lbl in intersections["label"]
        for part in lbl.split(" / ")
        if not part.strip().startswith("Knotenpunkt")
    }
)
street_names_set = set(street_names)
 
 
def get_node_streets(node_id: int) -> list: # return the known street names that belong to *node_id*.
    rows = intersections[intersections["node_id"] == node_id]
    if rows.empty:
        return []
    lbl = rows.iloc[0]["label"]
    return [p.strip() for p in lbl.split(" / ") if p.strip() in street_names_set]


# ─────────────────────────────────────────────────────────────────────────────
# label lookups
# ─────────────────────────────────────────────────────────────────────────────
 
label_to_id: dict[str, int] = dict(
    zip(intersections["unique_label"], intersections["node_id"].astype(int))
)
id_to_label: dict[int, str] = {v: k for k, v in label_to_id.items()}
labels = list(label_to_id.keys())

# ─────────────────────────────────────────────────────────────────────────────
# session state
# ─────────────────────────────────────────────────────────────────────────────
if "light_states" not in st.session_state:
    rng = np.random.default_rng(42)
    st.session_state.light_states = {
        int(r["node_id"]): PHASES[int(rng.integers(0, 3))]
        for _, r in intersections.iterrows()
    }

# keep in sync if cache was invalidated
for _, _row in intersections.iterrows():
    _nid = int(_row["node_id"])
    if _nid not in st.session_state.light_states:
        st.session_state.light_states[_nid] = "red"

if "selected_node" not in st.session_state:
    st.session_state.selected_node = int(intersections.iloc[0]["node_id"])

if "auto_cycle" not in st.session_state: ## Should the autocycle be on by default?
    st.session_state.auto_cycle = True

if "map_key_version" not in st.session_state:
    st.session_state.map_key_version = 0

# "selected_label" is the key= for the intersection selectbox.
# Using key= instead of index= means Streamlit owns the widget value in session
# state; a single dropdown pick always registers on the first click.
if "selected_label" not in st.session_state:
    st.session_state["selected_label"] = id_to_label.get(
        st.session_state.selected_node, labels[0]
    )

if "gw" not in st.session_state and street_names:
    _init_streets = get_node_streets(st.session_state.selected_node)
    st.session_state["gw"] = _init_streets[0] if _init_streets else street_names[0]

if "pending_map_click" in st.session_state:
    _clicked_nid = int(st.session_state.pop("pending_map_click"))
    st.session_state.selected_node = _clicked_nid
    st.session_state["selected_label"] = id_to_label.get(_clicked_nid, labels[0])
    st.session_state.map_key_version += 1  
    _click_streets = get_node_streets(_clicked_nid)
    if _click_streets:
        st.session_state["gw"] = _click_streets[0]
 

# ─────────────────────────────────────────────────────────────────────────────
# helper (traffic-light widget)
# ─────────────────────────────────────────────────────────────────────────────

def traffic_light_html(phase: str) -> str:
    off = {"red": "#3a0a0a", "yellow": "#2e2800", "green": "#002e10"}
    on_ = {"red": "#ee1c1c", "yellow": "#ffc800", "green": "#00dd44"}
    glow = {
        "red":    ("filter:drop-shadow(0 0 11px #ff2222cc)", "", ""),
        "yellow": ("", "filter:drop-shadow(0 0 11px #ffc800cc)", ""),
        "green":  ("", "", "filter:drop-shadow(0 0 11px #00ff44cc)"),
    }
    lights = [
        (on_["red"]    if phase == "red"    else off["red"],    glow[phase][0]),
        (on_["yellow"] if phase == "yellow" else off["yellow"], glow[phase][1]),
        (on_["green"]  if phase == "green"  else off["green"],  glow[phase][2]),
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


# ─────────────────────────────────────────────────────────────────────────────
# layout
# ─────────────────────────────────────────────────────────────────────────────
col_map, col_ctrl = st.columns([3, 1])


# control panel
with col_ctrl:
    st.markdown("### Steuerung")
    chosen_label = st.selectbox(
        "Kreuzung auswählen",
        options=labels,
        key="selected_label",
    )
    sel_id = label_to_id[chosen_label]
    st.session_state.selected_node = sel_id

    _sel_streets = get_node_streets(sel_id)
    if _sel_streets:
        st.session_state["gw"] = _sel_streets[0]

    cur_phase = st.session_state.light_states.get(sel_id, "red")

    left_col, right_col = st.columns([1, 1])

    # traffic light visual
    with left_col:
        st.markdown(traffic_light_html(cur_phase), unsafe_allow_html=True)

    # phase buttons
    with right_col:
        if st.button("**Rot**", use_container_width=True, type="primary" if cur_phase == "red" else "secondary"):
            st.session_state.light_states[sel_id] = "red"
            st.rerun()
            
        if st.button("**Gelb**", use_container_width=True, type="primary" if cur_phase == "yellow" else "secondary"):
            st.session_state.light_states[sel_id] = "yellow"
            st.rerun()
            
        if st.button("**Grün**", use_container_width=True, type="primary" if cur_phase == "green" else "secondary"):
            st.session_state.light_states[sel_id] = "green"
            st.rerun()


    # Green wave 
    st.markdown("### Grüne Welle")

    gw_choice = st.session_state.get("gw", "")
    if gw_choice:
        if st.button("Aktivieren", use_container_width=False, type="primary"):
            for _, row in intersections.iterrows():
                if gw_choice in row["label"]:
                    st.session_state.light_states[int(row["node_id"])] = "green"
            st.rerun()

_gw_street = st.session_state.get("gw", "")
gw_node_ids: set = {
    int(row["node_id"])
    for _, row in intersections.iterrows()
    if _gw_street and _gw_street in row["label"]
}

    
# interactive map
with col_map:
    scatter_data = []
    for _, row in intersections.iterrows():
        node_id = int(row["node_id"])
        phase = st.session_state.light_states.get(node_id, "red")
        scatter_data.append(
            {
                "node_id":     node_id,
                "lat":         float(row["lat"]),
                "lon":         float(row["lon"]),
                "label":       row["unique_label"],
                "phase_label": PHASE_LABELS[phase],
                "color":       PHASE_COLORS[phase],
                # Visual priority: selected > green-wave > default
                "radius": (
                    18 if node_id == st.session_state.selected_node else
                    13 if node_id in gw_node_ids else
                    8
                ),
                # White ring = selected node; transparent otherwise
                "line_color": (
                    [0, 0, 0, 255] if node_id == st.session_state.selected_node
                    else [0, 0, 0, 0]
                ),
            }
        )

    scatter_data.sort(
        key=lambda d: 1 if d["node_id"] == st.session_state.selected_node else 0
    )

    deck = pdk.Deck(
        layers=[
            # subtle road network background
            pdk.Layer(
                "PathLayer",
                data=edge_paths,
                get_path="path",
                get_color=[65, 65, 95, 100],
                get_width=3,
                width_scale=1,
                width_min_pixels=1,
            ),
            # intersection dots coloured by phase
            pdk.Layer(
                "ScatterplotLayer",
                id="intersections", 
                pickable=True,
                auto_highlight=False,
                get_position=["lon", "lat"],
                get_fill_color="color",
                get_radius="radius",
                radius_scale=3,
                radius_min_pixels=5,
                radius_max_pixels=24,
                stroked=True, 
                get_line_color="line_color", 
                line_width_min_pixels=2,
            ),
        ],
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
                "<div style='font-family:system-ui,sans-serif;font-size:13px;"
                "line-height:1.6;'>"
                "<b style='font-size:14px;'>{label}</b><br/>"
                "Phase: {phase_label}"
                "</div>"
            ),
            "style": {
                "backgroundColor": "rgba(15,15,28,0.93)",
                "color": "#ffffff",
                "padding": "8px 14px",
                "borderRadius": "8px",
                "boxShadow": "0 2px 12px rgba(0,0,0,.4)",
            },
        },
    )

    st.caption("Kreuzung in der Karte anklicken, um sie auszuwählen")
 
    try:
        event = st.pydeck_chart(
            deck,
            width="stretch",
            key=f"ampel_map_{st.session_state.map_key_version}",
            on_select="rerun",
            selection_mode="single-object",
        )
    except TypeError:
        # Fallback for older Streamlit versions that don't support on_select.
        st.pydeck_chart(deck, width="stretch", key=f"ampel_map_{st.session_state.map_key_version}",)
        event = None
 
    # map-node click
    # Store the clicked node as "pending" and rerun so the control panel (which is already rendered) picks up the change on the next pass
    if event is not None and hasattr(event, "selection") and hasattr(event.selection, "objects"):
        _sel_objs = event.selection.objects
        _clicked_list = (
            (_sel_objs.get("intersections") or next(iter(_sel_objs.values()), []))
            if _sel_objs else []
        )
        if _clicked_list:
            _clicked_nid = int(_clicked_list[0].get("node_id", -1))
            if _clicked_nid != -1 and _clicked_nid != st.session_state.selected_node:
                st.session_state["pending_map_click"] = _clicked_nid
                st.rerun()


st.markdown("### Alle Ampeln")
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    # metrics
    states = list(st.session_state.light_states.values())
    n_total = len(states)
    n_green  = states.count("green")
    n_yellow = states.count("yellow")
    n_red    = states.count("red")

    col_green, col_yellow, col_red = st.columns([1, 1, 1])
    with col_green:
        st.metric("🟢 Grün",   n_green,  delta=f"{n_green/n_total*100:.0f} %", delta_arrow="off", delta_color="green")
    with col_yellow:
        st.metric("🟡 Gelb",   n_yellow, delta=f"{n_yellow/n_total*100:.0f} %", delta_arrow="off", delta_color="yellow")
    with col_red:
        st.metric("🔴 Rot",    n_red,    delta=f"{n_red/n_total*100:.0f} %", delta_arrow="off", delta_color="red")

with col2:
    # controls
    bc1, bc2 = st.columns(2)
    with bc1:
        all_red = all(state == "red" for state in st.session_state.light_states.values())
        if st.button("Alle Ampeln Rot", use_container_width=True, type="primary" if all_red else "secondary"):
            for k in st.session_state.light_states:
                st.session_state.light_states[k] = "red"
            st.rerun()
    with bc2:
        all_green = all(state == "green" for state in st.session_state.light_states.values())
        if st.button("Alle Ampeln Grün", use_container_width=True, type="primary" if all_green else "secondary"):
            for k in st.session_state.light_states:
                st.session_state.light_states[k] = "green"
            st.rerun()

    auto_lbl = "⏹ Auto-Zyklus" if st.session_state.auto_cycle else "▶ Auto-Zyklus"
    if st.button(
        auto_lbl,
        use_container_width=True,
        type="primary" if st.session_state.auto_cycle else "secondary",
    ):
        st.session_state.auto_cycle = not st.session_state.auto_cycle
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# auto cycle
# ─────────────────────────────────────────────────────────────────────────────

if st.session_state.auto_cycle:
    time.sleep(1.8)
    rng = np.random.default_rng()
    for nid in list(st.session_state.light_states.keys()):
        if rng.random() > 0.72:   # ~28 % of lights advance per tick
            curr = st.session_state.light_states[nid]
            nxt  = PHASES[(PHASES.index(curr) + 1) % len(PHASES)]
            st.session_state.light_states[nid] = nxt
    st.rerun()
