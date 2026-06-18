import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# Beschreibung & Simulationsparameter je Straßentyp
# ─────────────────────────────────────────────────────────────────────────────
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

# values as used in the data simulation (max_speed, max_capacity, importance)
HIGHWAY_SIM_PARAMS = {
    "motorway":       (120, 1900, "sehr hoch"),
    "motorway_link":  (80,  1200, "hoch"),
    "trunk":          (80,  1500, "sehr hoch"),
    "trunk_link":     (60,  1000, "hoch"),
    "primary":        (50,  900,  "hoch"),
    "primary_link":   (45,  650,  "hoch"),
    "secondary":      (45,  700,  "hoch"),
    "secondary_link": (40,  550,  "mittel"),
    "tertiary":       (40,  550,  "mittel"),
    "tertiary_link":  (35,  420,  "mittel"),
    "unclassified":   (35,  360,  "mittel"),
    "residential":    (30,  260,  "gering"),
    "service":        (20,  110,  "gering"),
    "living_street":  (12,  120,  "gering"),
}
HIGHWAY_SIM_DEFAULT = (30, 240, "gering")


@st.dialog("Straßentypen & Methodik", width="large")
def show_info_dialog(present_highway_types: list[str]) -> None:    
    # html for left col (table of all streets + descriptions in this map))
    rows = "".join(
        f"<tr><td style='font-weight: 600; padding: 6px 8px; vertical-align: top; white-space: nowrap; width: 30%;'>{ht}</td>"
        f"<td style='padding: 6px 8px; vertical-align: top;'>{HIGHWAY_DESCRIPTIONS.get(ht, 'Keine Beschreibung verfügbar')}"
        f"<br><span style='opacity: 0.7; font-size: 0.75rem; font-weight: 400;'>"
        f"Tempolimit (Standard): {speed} km/h &nbsp;·&nbsp; "
        f"Kapazität: {cap} Fz/h pro Spur &nbsp;·&nbsp; "
        f"Verkehrsbedeutung: {imp}</span></td></tr>"
        for ht in sorted(present_highway_types)
        for speed, cap, imp in [HIGHWAY_SIM_PARAMS.get(ht, HIGHWAY_SIM_DEFAULT)]
    )

    col_left, col_right = st.columns(2, gap="large")

    with col_left:
        st.markdown("<b style='font-size: 1.25rem; display: block; margin-bottom: 12px;'>Straßentypen auf dieser Karte</b>", unsafe_allow_html=True)
        st.markdown(
            f"<table style='width: 100%; border-collapse: collapse; font-size: 0.83rem; line-height: 1.6;'>{rows}</table>",
            unsafe_allow_html=True
        )

    with col_right:
        st.markdown("<b style='font-size: 1.25rem; display: block; margin-bottom: 12px;'>So werden die Werte berechnet</b>", unsafe_allow_html=True)
        
        st.markdown("**Fahrzeuganzahl**")
        st.write(
            "Gibt an, wie viele Fahrzeuge sich im ausgewählten Zeitraum "
            "auf einem Straßenabschnitt befinden. Der Wert berücksichtigt "
            "Tageszeit, Straßentyp sowie den Einfluss umliegender "
            "Verkehrsschwerpunkte."
        )
        
        st.markdown("**Durchschnittsgeschwindigkeit**")
        st.write(
            "Die mittlere Fahrgeschwindigkeit der Fahrzeuge auf einem "
            "Abschnitt im gewählten Stundenzeitraum im Vergleich mit dem dort geltenden Tempolimit. "
            "Je stärker die Geschwindigkeit abweicht, desto höher "
            "ist das angezeigte Stau-Level."
        )
        
        st.markdown(
            """
            **Weitere Einflussfaktoren auf das Ergebnis**
            - Fahrspuren & Streckenlänge
            - Tageszeit & Wochentag
            - Pendlerrouten
            - Rückstau benachbarter Abschnitte
            - Vorfälle
            """
        )


def render_info_overlay(present_highway_types: list[str] | None, key: str = "info_overlay") -> None:
    if not present_highway_types:
        return

    with st.sidebar:
        # Ein einfacher, statischer Button reicht aus, da das Schließen vom Dialog übernommen wird
        if st.button("Wo kommen die Daten her?", width="stretch", key=f"{key}_btn"):
            show_info_dialog(present_highway_types)