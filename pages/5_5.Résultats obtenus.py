from pathlib import Path
import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd


st.title("Résultats obtenus")


tabs_title = ["🗃Texte uniquement", "🗃Image uniquement", "🗃🗃Texte & image"]
tab1, tab2, tab3 = st.tabs(tabs_title)

# TAB partie texte
with tab1:
    st.text("Résultats texte")


with tab2:
    st.text("Résultats image")


with tab3:
    st.text("Résultats texte et image")
    