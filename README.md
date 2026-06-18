# Struktur dieses Repos
## GUI
- GUI via Streamlit
- Karten-Rendering via pydeck
- Kartendaten via osmnx

### relevante Dateien
`GUI_backend.py` (Backend der Streamlit Site; Navigation, Logo, run site)  
`main_page.py` (Startseite mit allgemeinen Infos)  
`pg1_hist_data.py` (Karte inkl farbliche Stau-Markierungen und Daten der letzten 3 Monate als Tabelle)  
`pg2_live_data.py` (Karte inkl farbliche Stau-Markierungen mit Pseudo-Live-Daten über 5 Stunden)  
`pg3_forecast.py` (Karte inkl farbliche Stau-Markierungen mit den prognostizierten Verkehrsdaten der nächsten 8 Stunden)  
`pg4_traffic_lights_manual.py` (interaktive Karte mit Pseudo-Live-Ampelschaltungen; manuelles Eingreifen in jeder Kreuzung möglich)  
`pg5_traffic_lights_AI.py` (Karte inkl farbliche Stau-Markierungen und Markierungen der von der KI aufgrund der Prognose geänderten Ampelphasen; 2 mögliche Kartenansichten, Daten der Änderungen als Tabellen)  
`info_overlay.py` (Overlay im Sidebar von pg1, pg2 und pg3, erläutert die Daten im tooltip der Karten)  


## Datensimulation
- statische Simulation der letzten 3 Monate (23.03.2026 00:00 bis 23.06.2026 09:00)

### relevante Dateien
`simul_data.py` (Datensimulierung auf Basis von Tageszeit/-kategorie, Straßenart/-länge/-lage/-kapazität,...; Hotspots und vordefinierte Routen für realistische Daten)


## XGB-Prognosemodell
- Prognose erfolgt einmalig bei Ausführung für +8 Stunden nach Ende der Eingabedaten
- sehr gute Metriken, da XGB die simulierten Daten sehr gut vorhersagen kann

### relevante Dateien
`xgb_forecast.py` (Prognose via XGBoost)  
`xgb_metrics.txt` (Berechnete Metriken des XGB-Modells bei der neusten Ausführung)


## Datenbanken
- Upload via Git LFS, damit Streamlit Zugriff hat
- Statische Daten, keine regelmäßige Aktualisierung

### relevante Dateien
`traffic_data_darmstadt_mitte.db` (Enthält eine Tabelle: `traffic_darmstadt` mit einem Eintrag für jede Stunde und jedes Segment in Darmstadt-Mitte)  
`traffic_forecasts.db` (exakt gleiche Struktur wie die simulierten Daten)


# Funktionsweise
## Streamlit Website
`https://darmstadt-verkehrsprognosen.streamlit.app/`
- bei langer Inaktivität geht die Website in den Ruhemodus --> einfach auf "Aufwecken" klicken

## lokale Ausführung
1) alle Pakete in `requirements.txt` installieren
2) optional (nur notwendig, wenn die .db-Dateien nicht schon existieren):  
    2.1) `simul_data.py` ausführen --> `traffic_data_darmstadt_mitte.db` wird erstellt  
    2.2) `xgb_forecast.py` ausführen --> `traffic_forecasts.db` wird erstellt  
3) Terminal im aktuellen Ordner öffnen
4) Eingeben: `streamlit run GUI_backend.py`
5) GUI läuft lokal auf `localhost`
6) Zum Beenden `Strg+C` im Terminal