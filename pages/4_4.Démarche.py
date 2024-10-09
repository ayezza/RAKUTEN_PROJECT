import os
from pathlib import Path
import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd


st.title("La démarche")
st.write(">La trajectoire suivante décrit les étapes de réalisation du projet et le jalons atteints jusqu'au jalon final, la sourenance :")
img_trajectoire = Image.open(os.path.join(os.getcwd(), "images", "trajectoire.png"))
st.image(img_trajectoire)

st.write("La démarche suivie se résumé ainsi :")
st.write("1. Effectuer l’analyse du problème posé dans sa globalité et tracer la trajectoire pour les étapes suivantes")
st.write("2. Analyser les datasets mis à notre disposition en tant que source de donnée : contenu (texte et images), \
    visualisation des données, la quantification des métriques et des indicateursstatistiques pertinents")
st.write("3. Réaliser toute la phase de preprocessing afin de rendre le dataset cible exploitable, \
    de meilleure qualité dans l’objectif d’obtenir des résultats performants (scoring, prédiction des données de tests ou de nouvelles données, temps d’exécution…")
st.write("4. S’agissant d’un problème de classification des produits selon leurs catégories, procéder aux expérimentations des modèles dans les grandes typologies, \
    à savoir la classification de ML (Machine Learning), de DEEP learning (CNN, RNN etc., voir utilisation des modèle pré-entrainé BERT, GLOVE etc.")

    