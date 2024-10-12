
import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import requests

# This is relative root directory af trai and test images
#IMAGES_ROOT = os.path.join(os.getcwd(), "images")
IMAGES_ROOT = r"https://www.anigraphics.fr/images"



st.title("Analyse et preprocessing")


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


def get_codes_df():
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
    df_codes = pd.DataFrame({'prdtypecode': list(cats.keys()), 'catégorie': list(cats.values())})
    return df_codes
    

pickles_apth = "./data/cleaned_data.pkl"
print("Reading from pickle file from " + f"{pickles_apth} ...")
df = pd.read_pickle(f"{pickles_apth}")

df_codes = get_codes_df()
df_codes['prdtypecode'] = df_codes['prdtypecode'].astype(int)
df['prdtypecode'] = df['prdtypecode'].astype(int)
df_with_cats = pd.merge(left=df, left_on='prdtypecode', right=df_codes, right_on='prdtypecode' ).sort_values(by='catégorie')

df_with_cats['prdtypecode'] = df_with_cats['prdtypecode'].astype('Int64')
df_with_cats.index = df_with_cats.index.astype('int64') 


#import io
#buffer = io.StringIO()
#df_with_cats.info(buf=buffer)
#s = buffer.getvalue()
#st.text(s)
#st.dataframe(df_with_cats)


# Configuration des tabs
tabs_title = ["🗃Texte & Image", "🗃Texte uniquement", "🗃Images uniquement"]
tab0, tab1, tab2 = st.tabs(tabs_title)

# TAB Analyse du texte
with tab0:
    st.header("Exploration du texte et images")
    st.write("L'objectif de l’exercice d’exploration est de :") 
    st.write("1- Identifier dans quelle mesure le contenu de la « désignation » et de la « description » peut se rapporter à un type de produit")
    st.write("2- Quel type de nettoyage des données doit être effectué pour que les données textuelles soient les plus pertinentes ?")
    st.write("3- Déterminer la sémantique derrière le code de type de produit en affichant des images aléatoires et les principaux mots-clés pour chaque code de type de produit, ce qui aiderait à une meilleure interprétation humaine des résultats au cours de l'expérimentation du modèle.")
    img_explore_txt_byprdcode = Image.open(os.path.join(os.getcwd(), "images", "explore-txt-prdcat.png"))
    st.image(img_explore_txt_byprdcode)

    st.header("Conclusions")
    st.divider()
    st.write("La combinaison de la « désignation » et de la « description » semble donner de meilleurs résultats en termes d'identification des caractéristiques liées au type de produit.")
    st.divider()
    st.write("Les termes (token) pouvant être ignorés :")
    st.write("* Stop words pour l’anglais et le français")
    st.write("* ponctuation")
    st.write("* vocabulaire des dimensions : cm, mm, hauteur, etc.")
    st.write("* vocabulaire des couleurs : blanc, gris, etc.")
    st.write("* Balises et encodage HTML")
    st.write("* Valeurs numériques")
    st.write("* Besoin de lemmatisation : processus de réduction d'un token à son lemme")
    st.divider()
    st.write("La liste des catégories descriptives a été établie comme suit :")
    st.dataframe(df_codes)


with tab1:
    
    img_analyse_graphe_1 = Image.open(os.path.join(os.getcwd(), "images",  "analyse_graphe_1.png"))
    img_analyse_graphe_2 = Image.open(os.path.join(os.getcwd(), "images",  "analyse_graphe_2.png"))
    img_analyse_graphe_3 = Image.open(os.path.join(os.getcwd(), "images",  "analyse_graphe_3.png"))
    img_analyse_graphe_4 = Image.open(os.path.join(os.getcwd(), "images",  "analyse_graphe_4.png"))
    
    #img_analyse_graphe_1 = Image.open(requests.get(IMAGES_ROOT +  "/"  + "analyse_graphe_1.png", stream=True).raw)
    #img_analyse_graphe_2 = Image.open(requests.get(IMAGES_ROOT +  "/"  + "analyse_graphe_2.png", stream=True).raw)
    #img_analyse_graphe_3 = Image.open(requests.get(IMAGES_ROOT +  "/"  + "analyse_graphe_3.png", stream=True).raw)
    #img_analyse_graphe_4 = Image.open(requests.get(IMAGES_ROOT +  "/"  + "analyse_graphe_4.png", stream=True).raw)
    
    
    st.header("Graphes de répartitions avant le cleanning")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.html("<h4><span  style='color:orange'>Répartition des catégories selon les decriptions associées :</span></h4>")
        st.image(img_analyse_graphe_1)
        st.write("1.	Pour certaines catégories (parties noires) les valeurs nulles sont majoritaires, ce qui les pénalise comparativement aux autres catégories. \
                Cela peut éventuellement avoir une conséquence dans la prédiction de ces catégories au profit des autres.")
        st.write("2.	Même en ignorant les valeurs nulles, un déséquilibre subsiste toutefois, comme on le remarque clairement dans le graphe en bas.") 
        st.write("3.	Les catégories qui représentent clairement des cas anormaux, voire aberrants comparativement au reste, \
            comme le **2583**, ce qui risque de le favoriser dans les prédictions par l’effet de l’**OVERFITTING**") 
        st.write("4.	Ce constat est le même concernant le code catégorie 2583 en ce qui concerne aussi la variable ‘designation’")
        st.write("5.	Trois catégories qui sortent du lot, **1920, 2522 et 2583** représentent un nombre important de lignes non nulles et dupliquées")

        

    with col2:
        
        st.html("<h4><span  style='color:orange'>Répartition des catégories selon les designations associées :</span></h4>")
        st.image(img_analyse_graphe_2)
        st.write("On note bien que malgré ces trois distinctions, le déséquilibre subsiste ! Il s'accentue même pour quelques catégories en bas de l’échelle, \
            comme **60, 1180, 1301, 1940 et 2220**. Autrement dit, ces catégories seront moins fournies en termes de texte.")
    
    with col3:
        st.html("<h4><span  style='color:orange'>Répartition des catégories selon les designations associées après le cleaning :</span></h4>")
        st.image(img_analyse_graphe_3)
        st.html("Les actions suivantes ont été menées pour donner la nouvelle répartition des catégories ci-dessous un peu mieux équilibrée que précédemment\
            <ul><li>Suppression des valeurs nulles</li><ul>\
            <ul><li>Suppression des doublons</li><ul>\
            <ul><li>Suppression des expressions n'apportant aucune valeur sémantique relative au produit</li><ul>\
            <ul><li>Ajout d'une variable descriptive des 27 catgories à partir de l'analyse des images associées aux produits</li><ul>")
        
    
    st.html("<h4><span style='color:orange'>Cartographie en cloud des mots après le cleaning :</span></h4>") 
    st.html("<p>Les mots en fonction de leur taille dans le cloud, révèlent leurs fréquence dans les deux variables explicatives combinées,\
            la désignation des produits et la description associée. D'une manière indirecte, les mots mis plus en avant révèlent la catégorie des produits\
            la plus dominante en terme de description comme les mot <span style='color: red'><strong>jeu, enfant, sac, piscine...</strong></span></p>")
    st.image(img_analyse_graphe_4)
    
        
    st.html("<hr")    
    
    
    st.header("Graphes de répartitions après le cleanning en live")
    #   Load some graphs in live
    def drawBtn_1():
        btn_load_graphs = st.button("Charger les graphes",  type="primary" )
        if btn_load_graphs:
            load_graphs()
            st.text("Les graphes ont été chargés !")
    
    def load_graphs():  
        if isFileExist(pickles_apth):
            col1, col2, col3 = st.columns(3)
            
            sns.set_theme(rc={'figure.figsize': (10, 7)})
            fig, ax = plt.subplots(nrows= 1, ncols= 1)
            
            with col1:
                st.write(">Distribution de la longueur de la variable **desi_desc**")
                g1 = sns.histplot(x=df['desi_desc'].str.split().map(lambda x: len(x)), ax=ax, kde=True, bins=range(0, 400))
                g1.set_xlim(0,400)
                g1.set_xlabel("Longueur du texte 'desi_desc'")
                g1.set_ylabel("Nombre de 'desi_desc'")
                g1.set_title("Distribution de la longueur de la variable 'desi_desc'")
                g1.set_gid(True)
                st.pyplot(g1.figure)
                st.write(">Un nombre important de lignes possèdent une longeurs importante du texte **desi_desc**. \
                    Le calcul des quartiles standards Q1, Q2 et Q3, révèleront l'ampleur des outiliers sur la base \
                    de l'indicateur de dispertion supérieure : **Q3 + 1.5*IQR**. Voir graphe à droite.")
            
            with col2:
                st.write(">Distribution en BOXPLOT (Moustache) de la longueur de la variable **desi_desc**")
                df['desc_length'] = df['desi_desc'].apply(lambda x: len(str(x)))
                g2 = sns.boxplot(data=df, y='desc_length', ax=ax, hue='prdtypecode', gap=1.5, palette='pastel')
                g2.set_ylabel("Longueur du texte")
                g2.set_title("Distribution de la longueur de la variable 'desi_desc'")
                st.pyplot(g2.figure)
                st.write(">La majorité des catégories possédent des outiliers supérieurs a des ampleurs qui ne sont pas au même niveau.\
                    Cela est la conséquence de leur nombre et de la longeur du texte aussi")
            
            with col3:
                st.write(">Distribution du nombre de produits par catégorie")
                g3 =sns.countplot(data=df_with_cats,  x='prdtypecode',  orient='v', palette='Spectral')
                g3.set_title("Distribution en nombre de produits par catégorie")
                g3.set_xlabel('Catégories')
                g3.set_ylabel('Nombre de produits')
                ax.tick_params(axis='x', rotation=90)
                plt.legend(loc='lower left')
                #g3.set_xticklabels(labels=df_with_cats["catégorie"].unique())
                st.pyplot(g3.figure)
                st.write(">On voit clairement un déséquilibre dans cette répartition qui peut être une \
                    source d’OVERFITING ou de résultats de prédiction erronées en faveur des catégories \
                        dominantes comme la **2583** qui présente un cas aberrant ! ")
            
            #st.bar_chart(df_with_cats, x="catégorie", y="prdtypecode", color="catégorie", x_label ="Catégories", y_label="Nombre de produits", stack=False)
            
        else:
            st.html("Fichier PICKLE introuvable ici : " + pickles_apth)

    # action button
    drawBtn_1()


# TAB Analyse des images"
with tab2:
    img_explore_images_1 = Image.open(os.path.join(os.getcwd(), "images", "freq_img_taille_bits.png"))
    img_explore_images_2 = Image.open(os.path.join(os.getcwd(), "images", "boxplot_img_taille_bits.png"))
    img_explore_images_3 = Image.open(os.path.join(os.getcwd(), "images", "anova_test_img_taille_bits.jpg"))
    
    #img_explore_images_1 = Image.open(requests.get(IMAGES_ROOT +  "/"  + "freq_img_taille_bits.png", stream=True).raw)
    #img_explore_images_2 = Image.open(requests.get(IMAGES_ROOT +  "/"  + "boxplot_img_taille_bits.png", stream=True).raw)
    #img_explore_images_3 = Image.open(requests.get(IMAGES_ROOT +  "/"  + "anova_test_img_taille_bits.jpg", stream=True).raw)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.header("Fréquences des tailles des images (en bits)")
        st.image(img_explore_images_1)
        st.write("Ce graphe exprime la fréquence des images en fonction de leur poids, une piste que \
            nous choisissons d'explorer. On observe une distribution gaussienne, la **plus grande partie des images \
            pèse environ 20 000 bits, soit 20 kilobit**")


    with col2:
        st.header("Graphe boxplot des tailles")
        st.image(img_explore_images_2)
        st.write("Ce graphe Exprime la distribution de la taille des images en fonction des catégories \
                 médiane, écarts interquartiles et outliers. **Nous remarquons que cette distribution est assez disparate, \
                nous avons donc de bonnes raisons de penser que la taille des images influe sur la catégorie**")

    with col3:
        st.header("Test ANOVA sur les tailles des images")
        st.image(img_explore_images_3)
        st.write("Nous réalisons un test ANOVA (qui sert à savoir si plusieurs groupes ont des \
                différences significatives entre eux) avec les hypothèses suivantes:")
        st.write("> H0 : La taille des images n'a pas d'influence sur la catégorie")
        st.write("> H1 : La taille des images a une influence sur la catégorie")
        st.write("Au vu des résulats (**p-value très inférieure à 0.05**) et **F-stat très élevée (+ de 339)**, on peut rejeter \
                l'hypothèse H0 au profit de la H1 : la taille des images a une influence sur la catégorie. \
                C'est donc une feature qui pourrait éventuellement nous servir par la suite pour catégoriser les images, \
                mais nous n'en aurons pas besoin au final car les features les plus évidentes (valeurs de pixels des img) suffiront.")

    st.html("<hr")

  
    img_analyse_images_1 = Image.open(os.path.join(os.getcwd(), "images",  "exemple_preprocess_baseline.png"))
    img_analyse_images_2 = Image.open(os.path.join(os.getcwd(), "images",  "exemple_preprocess_deeplearning.png"))
    img_analyse_images_3 = Image.open(os.path.join(os.getcwd(), "images",  "exemple_preprocess_generique.png"))
    
    #img_analyse_images_1 = Image.open(requests.get(IMAGES_ROOT +  "/"  + "exemple_preprocess_baseline.png", stream=True).raw)
    #img_analyse_images_2 = Image.open(requests.get(IMAGES_ROOT +  "/"  + "exemple_preprocess_deeplearning.png", stream=True).raw)
    #img_analyse_images_3 = Image.open(requests.get(IMAGES_ROOT +  "/"  + "exemple_preprocess_generique.png", stream=True).raw)
    
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.header("Preprocessing générique")
        st.image(img_analyse_images_1)
        st.write("Un exemple d'images preprocessées qui pourront nous servir pour tous les modèles (pas de réduction de taille, pas \
            d'altération de la qualité, 1000 images par catégorie et transfos)")
        
    with col2:
        st.header("Preprocess deep-learning")
        st.image(img_analyse_images_2)
        st.write("Un exemple d'images preprocessées pour les modèles deeplearning (la library utilisée est \
            **keras**, réduction taille en 224x224 et transfos")
   
    with col3:
        st.header("Preprocessing baseline")
        st.image(img_analyse_images_3)
        st.write("Un exemple_preprocess_baseline : un exemple d'images preprocessées pour les modèles baseline \
            (taille réduite, niveaux de gris et transfos")
   

    
    st.html("<span  style='color:orange; font-size: 24px;'>Nous avons effectué 3 types de preprocessing :</span>")
    
    st.write("1. **Un preprocessing générique**: utilisable par tout type de modèles. Il contient 1000 images par catégorie, et \
    30% des images ont été **augmentées** (rotations, zoom, etc.) pour diversifier le dataset. Leur taille est inchangée  \
    (500x500) et pourra par la suite être adaptée en fonction des modèles (voir **exemple_preprocess_generique**")
    
    st.write("2. **Un preprocessing pour les modèles de Deep Learning** : les modèles de Deep Learning sont inclues dans des  \
    librairies qui contiennent leurs propres fonctions de preprocessing (voici un exemple de dataset image  \
    preprocessé par keras de Tensorflow: **exemple_preprocess_deeplearning**). ce preprocessing sera adapté au cas par cas en fonction  \
    des modèles.")
    
    st.write("3. **Un preprocessing pour les modèles baseline** : Nous allons entraîner par la suite 2 types de modèles : des \
    modèles deep learning (plus complexes) et des modèles baseline (plus simples).  \
    Pour les modèles baseline, nous avons réduit les images en 64x64 pixels (au lieu de 500x500). \
    Nous les avons passées en niveaux de gris. (voir graphe **exemple_preprocess_baseline**")


