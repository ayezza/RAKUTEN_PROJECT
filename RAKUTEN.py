from pathlib import Path
import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import os

#from pages import page1, page2, page3  # Importer les modules de pages



st.set_page_config(
    page_title="Projet RAKUTEN",
    #page_icon="🚀",
    layout='wide'
)


st.title("Accueil - Projet RAKUTEN")
images_url = "https://github.com/DataScientest-Studio/jul24_bds_rakuten/tree/main/src/streamlit/images"
logo = Image.open(os.path.join(os.getcwd(), "images", "rakuten.png"))
st.image(image=logo)


img_logo= Image.open(os.path.join(os.getcwd(), "images", "logo_datascientest.png"))
with st.sidebar:
    st.image(img_logo)
    st.success("Sélectionner les rubriques ci-dessus :")
    
# st.sidebar.success("Sélectionner les rubriques ci-dessus :")

st.html("<h2>Session<h2>\
    <p>Juil. 2024 - Promotion BOOTCAMP DS</p>")
st.html("<h2>Mentor<h2>\
    <p>Eliott Douieb</p>")
st.html("<h2>Equipe projet</h3>\
    <ul><li>Louis VALENTIN</li><li>Souleymane TOURE</li><li>Ouissam GOUNI</li><li>Abdel YEZZA</li></ul>")

st.html('<a href="https://github.com/DataScientest-Studio/jul24_bds_rakuten" target="_blank">Espace GITHUB du projet</a>')    
