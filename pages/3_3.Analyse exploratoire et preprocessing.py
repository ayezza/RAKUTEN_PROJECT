from pathlib import Path
import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


def isFileExist(fileFullPath):
    if fileFullPath is not None:
        if os.path.isfile(fileFullPath):
            if os.path.exists(fileFullPath):
                return True
            else:
                return False
        else:
            return False
    else:
        return False


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
    
    
    #   Load some graphs in live
    st.html("<h1><span  style='color:orange'>Graphes de distribution</span></h1>")
    def drawBtn_1():
        btn_load_graphs = st.button("Charger les graphes",  type="primary" )
        if btn_load_graphs:
            load_graphs()
            st.text("Les graphes ont été chargés !")
    
    def load_graphs():
        pickles_apth = "./data/cleaned_data.pkl"
        if isFileExist(pickles_apth):
            print("Reading from pickle file from " + f"{pickles_apth} ...")
            df = pd.read_pickle(f"{pickles_apth}")
            sns.set_theme(rc={'figure.figsize': (10, 10)})
            fig, ax = plt.subplots(nrows= 1, ncols= 1)
            g1 = sns.histplot(x=df['desi_desc'].str.split().map(lambda x: len(x)), ax=ax, kde=True, bins=range(0, 400))
            
            g1.set_xlim(0,400)
            g1.set_xlabel("Longueur du texte 'desi_desc'")
            g1.set_ylabel("Nombre de 'desi_desc'")
            g1.set_title("Distribution de la longueur de la variable 'desi_desc'")
            g1.set_gid(True)
            st.pyplot(g1.figure)
            
            df['desc_length'] = df['desi_desc'].apply(lambda x: len(str(x)))
            g2 = sns.boxplot(data=df, y='desc_length', ax=ax, hue='prdtypecode', gap=1.5, palette='pastel')
            g2.set_ylabel("Longueur du texte")
            g2.set_title("Distribution de la longueur de la variable 'desi_desc'")
            st.pyplot(g2.figure)

        else:
            st.html("Fichier PICKLE introuvable ici " + pickles_apth)

    drawBtn_1()
    
    
with tab2:
    st.html("<h1><span  style='color:orange'>Echantillons d'images par code catégorie des produits</span></h1>")
    def drawBtn_2():
        btn_load_images = st.button("Charger les échantillons",  type="primary" )
        if btn_load_images:
            load_images()
            st.text("Les images ont été chargées !")
        
    def load_images():
        cats = {
            '10' : 'Livres anciens / occasion',
            '40' : 'Jeux vidéos anciens, équipement',
            '50' : 'Accessoires & produits dérivés gaming',
            '60' : 'Consoles de jeu',
            '1140' : 'Figurines',
            '1160' : 'Cartes de jeu',
            '1180' : 'Figurines & Jeux de Société',
            '1280' : 'Jeux & jouets pour enfants',
            '1281' : 'Jeux de société',
            '1300' : 'Modélisme',
            '1301' : 'Vêtements bébé et jeux pour la maison',
            '1302' : 'Jeux & jouets d\'extérieur pour enfants',
            '1320' : 'Jouets & accessoires pour bébé',
            '1560' : 'Meubles d\'intérieur',
            '1920' : 'Linge de maison',
            '1940' : 'Alimentation & vaisselle',
            '2060' : 'Objets décoration maison',
            '2220' : 'Equipement pour animaux',
            '2280' : 'Journaux, revues, magazines anciens',
            '2403' : 'Livres, BD, magazines anciens',
            '2462' : 'Consoles, jeux et équipement occasion',
            '2522' : 'Papeterie',
            '2582' : 'Meubles d\'extérieur',
            '2583' : 'Equipement pour piscine',
            '2585' : 'Outillage intérieur / extérieur, tâches ménagères',
            '2705' : 'Livres neufs',
            '2905' : 'Jeux PC',
        }
        df = pd.DataFrame({'code': list(cats.keys()), 'catégorie': list(cats.values())})
        for code in df['code']:
            st.write("Catégorie : " + cats[code])
            st.image(Image.open(os.path.join(os.getcwd(), "images", "code-" + str(code) + ".png")))
        
    drawBtn_2()        
    
    