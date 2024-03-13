import streamlit as st
import pandas as pd
import numpy as np
import os
import folium
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
from shapely.geometry import Point, Polygon
from folium.plugins import Fullscreen
import time
import base64
from PIL import Image
from streamlit_extras.switch_page_button import switch_page


def streamlit_settings(title, icon):
    st.set_page_config(page_title=title, page_icon=icon, layout="centered", initial_sidebar_state='collapsed')
    with open("src/styles/main.css") as f:
        st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)
    st.markdown(
        """<style>[data-testid="collapsedControl"] svg {height: 3rem;width: 3rem;}</style>""",
        unsafe_allow_html=True,
    )

def download_pdf():
    pdf_file_path = "Rapport.pdf"

    with open(pdf_file_path, "rb") as file:
        pdf_bytes = file.read()

    st.download_button(
        label="Last ned fullstendig rapport",
        data=pdf_bytes,
        help="Trykk på knappen for å laste ned rapport",
        file_name="Rapport.pdf",
        key="pdf-download",
    )


streamlit_settings(title="Energy Plan Zero, Kringsjå", icon="h")
st.title("Energiplan Kringsjå")
st.header("Innledning")
with st.sidebar:
    st.image('src/img/sio-av.png', use_column_width="auto")

st.write("""Asplan Viak har på oppdrag for SiO utarbeidet en mulighetsstudie 
         for energiforsyning til Kringsjå studentby. Formålet med studien er å 
         få et overordnet beslutningsgrunnlag for strategisk valg av bærekraftig energiforsyning i et 30 års perspektiv.""")

#download_pdf()

st.header("Anbefalinger")
st.image('src/img/tiltak.PNG', use_column_width="auto")

st.write("")
st.write("")

if st.button("Gå til kartapplikasjon"):
    switch_page("Kartapplikasjon")