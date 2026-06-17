#### DOKU: https://docs.streamlit.io/develop/api-reference 

import streamlit as st
import base64

# Define the pages
main_page = st.Page("main_page.py", title="Startseite", default=True)
page_1 = st.Page("pg1_hist_data.py", title="Historische Verkehrsdaten")
page_2 = st.Page("pg2_live_data.py", title="Live-Daten")
page_3 = st.Page("pg3_forecast.py", title="Verkehrsprognose")
page_4 = st.Page("pg4_traffic_lights_manual.py", title="Ampelschaltungssoftware")
page_5 = st.Page("pg5_traffic_lights_AI.py", title="KI-gestützte Ampelschaltung")

# Set up navigation
pg = st.navigation(pages=[main_page, page_1, page_2, page_3, page_4, page_5], position="sidebar")

st.markdown( # change font size of navigation bar
    """
    <style>
    div[data-testid="stSidebarNav"] ul[data-testid="stSidebarNavItems"] li a[data-testid="stSidebarNavLink"] span {
        font-size: 20px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# add company logo to sidebar
def get_image_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()
    
logo_base64 = get_image_base64("img/logo-removebg.png")

st.sidebar.markdown(
    f"""
    <div style="position: fixed; bottom: 20px; padding-bottom: 10px; display: flex; align-items: center; gap: 10px;">
        <p style="margin: 0;">powered by</p>
        <img src="data:image/png;base64,{logo_base64}" 
             style="height: 80px; width: auto; filter: drop-shadow(0px 0px 1px rgba(255, 255, 255, 1));">
    </div>
    """,
    unsafe_allow_html=True
)

# run this page
pg.run()