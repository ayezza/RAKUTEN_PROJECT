from pathlib import Path
import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd


st.title("Démo")




tabs_title = ["🗃Démo texte uniquement", "🗃Démo image uniquement", "🗃🗃Démo texte & image"]
tab1, tab2, tab3 = st.tabs(tabs_title)

# TAB partie texte
with tab1:
    st.text("Démo texte")


with tab2:
    st.text("Démo image")


with tab3:
    st.text("Démo texte et image")
    
    