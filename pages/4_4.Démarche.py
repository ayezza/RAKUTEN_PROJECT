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

st.write("<h4>La démarche suivie se résumé en 4 étapes :</h4>", 
         unsafe_allow_html= True)
st.html("1. <h6><strong>Effectuer l’analyse du problème posé dans sa globalité et tracer la trajectoire pour les étapes suivantes</strong></h6>")
st.html("2. <h6><strong>Analyser les datasets mis à notre disposition en tant que source de donnée</strong> : contenu (texte et images), \
    visualisation des données, la quantification des métriques et des indicateurs statistiques pertinents</h6>")
st.html("3. <h6><strong>Réaliser toute la phase de preprocessing</strong> afin de rendre le dataset cible exploitable, \
    de meilleure qualité dans l’objectif d’obtenir des résultats performants (scoring, prédiction des données de tests ou de nouvelles données, temps d’exécution…</h6>")
st.html("4. <h6>S’agissant d’un problème de classification des produits selon leurs catégories, <strong>procéder aux expérimentations des modèles dans les grandes typologies</strong>, \
    à savoir la classification de ML (Machine Learning), de DEEP learning (CNN, RNN etc., voire utilisation des modèles pré-entrainés comme BERT, GLOVE etc.</h6>")

    