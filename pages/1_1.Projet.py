from pathlib import Path
import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import os


st.title("Présentation du projet")
st.html("<h1><span  style='color:orange'>Contexte</span></h1>\
    Ce projet s’inscrit dans le cadre de la promotion de formation “DataScientist” de juillet 2024. \
    Il porte sur des données provenant de RAKUTEN Institut of technology et a fait l’objet d’un challenge \
    avec une publication des résultats des benchmarks.<br><br>Ces mêmes données ont été reprises dans le cadre \
    du projet de fin de formation dans l’objectif de réaliser au minimum un modèle de classification uni-modal et/ou multi-modal \
    de produits e-commerce Rakuten à partir de textes et images issus des données. Cette vitrine STEAMLIT présente \
    une synthèse des travaux réalisés par l’équipe projet, des résultats obtenus, \
    mais aussi les difficultés rencontrées et les challenges futurs ouvrant des opportunités \
    à des améliorations des modèles proposés ou nouveaux.")

st.html("<p>Le but du travail réalisé dans le cadre du projet est de :</p>")
with st.container(height=300):    
    st.html("<span style='color:green;text-align:center;font-size:32px;height:32px;background-color:#FFFFFF;'> \
    Fournir un moyen au travers des modèles, de <strong>prédire la catégorie recherchée d’un produit en fonction \
        soit d’un texte descriptif, une ou plusieurs images, voire les deux à la fois.</strong>\
    </span><br><br>\
    \
    <span style='color:orange;text-align:center;font-size:24px;height:24px;background-color:#FFFFFF;'>\
    Les entreprises, notamment celles exposant des sites e-commerce, ont un besoin grandissant \
    d’identifier les désirs de leurs clients et visiteurs en termes d’achat de produits bien précis ou proposer des promotions bien visées. \
    </span>")
    
st.html(" C’est au travers des modèles éprouvés que ces sociétés peuvent répondre d’une manière rapide et la plus précise possible, \
        sujet sur lequel répond partiellement le projet réalisé dans le cadre de notre formation accélérée.</p>")

st.html("<h1><span  style='color:orange'>Objectifs</span></h1>\
    <ol><li>Construire sur la base des jeux de données fournis,  plusieurs modèles en mesure de catégoriser tout produit à partir de données textuelles explicatives et/ou des images</li>\
    <li>Appliquer les dernières avancées en matière de modélisation de l’apprentissage machine et des méthodes de classifications multimodales</li>\
    <li>Conjuguer les modèles de machine learning avec ceux de DEEP learning, voire introduire des modèles pré-entraînés comme \
        <strong>BERT (Bidirectional Encoder Representations from Transformers)</strong> ou <strong>GLOVE \
            (Global Vectors for Word Representation - fournissant des poids des mots les plus proches)</strong>\ etc.</li>\
    <li>Tester les modèles élaborés sur des données actuelles issues des sites e-commerce aussi bien RAKUTEN que d'autres fournisseurs dans la même gamme de produits </li></ol>")


st.html("<h1><span  style='color:orange'>Démarche</span></h1>")
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
    à savoir la classification de ML (Machine Learning), de DEEP learning (CNN, RNN etc., voire utilisation des modèles pré-entrainés comme BERT, GLOVE etc.)</h6>")
