from pathlib import Path
import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import os


st.title("Analyse exploratoire et preprossing")


# Configuration de la barre latérale
tabs_title = ["🗃Partie texte", "🗃Partie images"]
tab1, tab2 = st.tabs(tabs_title)

with tab1:
    img_analyse_graphe_1 = Image.open(os.path.join(os.getcwd(), "images", "analyse_graphe_1.png"))
    img_analyse_graphe_2 = Image.open(os.path.join(os.getcwd(), "images", "analyse_graphe_2.png"))
    img_analyse_graphe_3 = Image.open(os.path.join(os.getcwd(), "images", "analyse_graphe_3.png"))
    img_analyse_graphe_4 = Image.open(os.path.join(os.getcwd(), "images", "analyse_graphe_4.png"))
    
    st.html("<h1><span  style='color:orange'>Répartition des catégories selon les decriptions associées :</span></h1>")
    st.image(img_analyse_graphe_1)
    
    st.html("<h1><span  style='color:orange'>Répartition des catégories selon les designations associées :</span></h1>")
    st.image(img_analyse_graphe_2)
    
    st.html("<h1><span  style='color:orange'>Répartition des catégories selon les designations associées :</span></h1>\
           Les actions suivantes ont été menées pour donner la nouvelle répartition des catégories ci-dessous un peu mieux équilibrée que précédemment\
           <ul><li>Suppression des valeurs nulles</li><ul>\
            <ul><li>Suppression des doublons</li><ul>\
            <ul><li>Suppression des expressions n'apportant aucune valeur sémantique relative au produit</li><ul>\
            <ul><li>Ajout d'une variable descriptive des 27 catgories à partir de l'analyse des images associées aux produits</li><ul>")
    st.image(img_analyse_graphe_3)
    
    st.html("<h1><span style='color:orange'>Cartographie en cloud des mots les plus utilisés :</span></h1>")
    st.html("<p>Les mots en fonction de leur taille dans le cloud, révèlent leurs fréquence dans les deux variables explicatives combinées,\
        la désignation des produits et la description associée. D'une manière indirecte, les mots mis plus en avant révèlent la catégorie des produits\
            la plus dominante en terme de description comme les mot <span style='color: red'><strong>jeu, enfant, sac, piscine...</strong></span></p>")
    st.image(img_analyse_graphe_4)
    
with tab2:
    img_analyse_graphe_1 = Image.open(os.path.join(os.getcwd(), "images", "analyse_graphe_1.png"))
    img_analyse_graphe_2 = Image.open(os.path.join(os.getcwd(), "images", "analyse_graphe_2.png"))
    