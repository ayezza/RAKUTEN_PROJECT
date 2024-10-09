import os
import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd


st.title("Résultats obtenus")


tabs_title = ["🗃Texte uniquement", "🗃Image uniquement", "🗃🗃Bimodal texte & image"]
tab1, tab2, tab3 = st.tabs(tabs_title)

# TAB partie texte
with tab1: 
    
    st.write("### Modèles Machine Learning (ML)")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("SVM", "Accuracy", "77%", "off")
    col2.metric("Logistic Regression ", "Accuracy", "80%", "normal")
    col3.metric("KNN", "Accuracy", "55%", "off")
    col4.metric("XGBOOST", "Accuracy", "78%", "normal")
    
    st.html("<hr>")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("SVM", "f1_score", "?%", "off")
    col2.metric("Logistic Regression ", "f1_score", "79%", "normal")
    col3.metric("KNN", "f1_score", "55%", "off")
    col4.metric("XGBOOST", "f1_score", "?%", "normal")
    
    st.html("<hr>")
     
    st.write("### Modèles Réseaux de Neurones (RNN/RNC)")
    col1, col2, col3 = st.columns(3)
    col1.metric("LSTM combiné avec Conv1D", "Accuracy", "79%", "normal")
    col2.metric("LSTM renforcé avec GLOVE", "Accuracy", "80%", "normal")
    col3.metric("CNN (Conv1D)", "Accuracy", "80%", "normal")
    
    st.html("<hr>")
    st.write("### Modèles ML (approche différente)")
    st.write("Les mots ne sont pas vectorisés, mais transformés en variables descriptive avec un nombre limité à 300.\
    Cette approche n'a pas été retenue pour des questions de performance d'évaluation des ptrédictions.")
    img_approche_ml = Image.open(os.path.join(os.getcwd(), "images", "image-2.png"))
    st.image(img_approche_ml)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("KNN", "accuracy", "90%", "normal")
    col2.metric("KNN avec une réduction PCA de 57% des variables", "accuracy", "87%", "normal")
    col3.metric("Random Forest", "accuracy", "91%", "normal")
    col4.metric("Logistic Regression", "accuracy", "89%", "normal")
    col5.metric("SVC (Support Vector)", "accuracy", "85%", "normal")
    
    
with tab2:
    st.header("Résultats obtenus")


with tab3:
    st.header("Résultats obtenus")
    