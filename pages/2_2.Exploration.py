from pathlib import Path
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import numpy as np
import pandas as pd
import os
import requests


#IMAGES_ROOT = os.path.join(os.getcwd(), "images")
IMAGES_ROOT = r"https://www.anigraphics.fr/images"


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
    
def load_data_from_pickle():
    pickles_apth = "./data/cleaned_data.pkl"
    if isFileExist(pickles_apth):
        print("Reading from pickle file from " + f"{pickles_apth} ...")
        df = pd.read_pickle(f"{pickles_apth}")
        df_codes = get_codes_df()
        df_codes['prdtypecode'] = df_codes['prdtypecode'].astype(int)
        df['prdtypecode'] = df['prdtypecode'].astype(int)
        df['productid'] = df['productid'].astype('int64')
        df['imageid'] = df['imageid'].astype('int64')
        df_with_cats = pd.merge(left=df, left_on='prdtypecode', right=df_codes, right_on='prdtypecode' ).sort_values(by='catégorie')
        st.write("dataset chargé")
        return df_with_cats

    else:
        st.write("Impossible de charger le dataset ! ")
        return None



st.title("Exploration")

df_with_cats = load_data_from_pickle()
if df_with_cats is None:
    st.write("Ouch ! Impossible de charger le dataset à partir du PICKLE.")

# Configuration de la barre latérale
tabs_title = ["🚀Jeu de données intial", "🚀Jeu de données cible", "🚀Exploration intéractive des images"]
tab1, tab2, tab3 = st.tabs(tabs_title)
cur_dir = os.getcwd()
st.write(cur_dir)

with tab1:
    # liste des images statiques
    #img_train_set = Image.open(os.path.join(os.getcwd(), "images", "train_set.png"))
    #img_test_set = Image.open(os.path.join(os.getcwd(), "images", "test_set.png"))
    #img_target_set = Image.open(os.path.join(os.getcwd(), "images", "target_set.png"))
    #img_schema_dataset = Image.open(os.path.join(os.getcwd(), "images", "schema_source_dataset.png"))
    
    img_train_set = Image.open(requests.get(IMAGES_ROOT +  "/"  + "train_set.png", stream=True).raw)
    img_test_set = Image.open(requests.get(IMAGES_ROOT +  "/"  + "test_set.png", stream=True).raw)
    img_target_set = Image.open(requests.get(IMAGES_ROOT +  "/"  + "target_set.png", stream=True).raw)
    img_schema_dataset = Image.open(requests.get(IMAGES_ROOT +  "/"  + "schema_source_dataset.png", stream=True).raw)
    
    
    
    st.html("<h3>⭐<span  style='color:orange'>Schéma des jeux de données d'entraînement et de test :</span></h3>")
    st.image(img_schema_dataset)
    st.html("\
        <ul><li>Deux fichiers CSV, le 1er <strong>X_train_update.csv</strong> est un le jeu de données d'entraînement et le 2ème <strong>  X_test_update.csv</strong>est celui de test </li>            \
            <li>Les deux fichiers CSV possèdent le même nombre de variables explicatives comme illustré ci-dessous</li>        \
            <li>Chaque ligne du CSV référence d'une manière unique le ID du produit et le ID de l'image représentant le produit</li>        \
        </ul>\
             ")
    
    st.html("<h3>⭐<span  style='color:orange'>Jeu de données d'entraînement :</span></h3>")
    st.image(image=img_train_set)
    st.write(">Les valeurs manquantes sont uniquement dans la colonne **description** de l'ordre de 35%")
    st.write(">Le taux des designations en doublon reste relativement faible 3%")
    st.write(">Quant au taux des descriptions en doublon de l'ordre de 45% est assez élevé du fait certainement du copier/coller pour les produits dans la même catégorie/univers !")
    
  
    st.write(">Remarque : Le jeu de données d'entraînement ne contienent pas les données relatives à la cible,\
        à savoir, les **catégories associées aux produits**.")
     
    img_shema_target = Image.open(os.path.join(os.getcwd(), "images", "shema_target.png"))
    st.html("<h3>⭐<span  style='color:orange'>Schéma de la variable à prédire (catégories associées aux produits) :</span></h3>")
    st.html("<p>La variable cible est formée de 27 catégories numériques. Une variable traduisant les catégories numériques \
        ci-dessous en catégories descirptives sur la base de la combinaison du texte et des images associées aux produits.</p>")
    st.image(img_shema_target)
    
    #st.html("<h3><span  style='color:orange'>List des catégories descriptives :</span></h3>")
    #st.write("On a rajouté une colonne descriptive de la cible, à savoir les 27 catégories des produits :")
    #df = get_codes_df()
    #st.dataframe(df)
    
    st.html("<h3>⭐<span  style='color:orange'>Jeu de données cible (target) pour la partie entraînement</span></h3>")
    st.write("le jeu de données target de la partie test possède le même schéma que celui d'entraînement.")
    st.write("Dimension du jeu de données target pour le test : (16453, 1)")
    st.image(img_target_set)
    
    st.html("<h4>⭐<span  style='color:orange'>Dataset cible</span></h4>")
    st.write(">#### Le CSV d'entraînement combiné avec le CSV de la cible (catégories des produits) sera notre dataset pour la modélisation")
    st.write(">#### Un CSV de test sans les catégories sera uniquement utilisé lors de tests des modèles développés")
    
with tab2:
    img_datset_cibe = Image.open(os.path.join(os.getcwd(), "images", "datset_cibe.png"))
    st.html("<h3>⭐<span  style='color:orange'>Schéma du jeu de données cible construit :</span></h3>")
    st.image(img_datset_cibe)
    
    st.html("<span style='font-size: 20px;'>Les deux variables <span style='color:orange'>'designation'</span> et <span style='color:orange'>'description'</span> ont été concténées \
        dans une variable <span style='color:orange'>'desi_desc'</span> qui devient l'unique variable explicative textuelle utilisée dans les modèles.\
        Le reste des variables explicatives ont été préservées.</span>")
    
    st.html("<h4>Aperçu du jeu de données augmenté, transformé et nettoyé</h4>")
    st.write(">Uniquement la nouvelle colonne **desi_desc** fusion de **designation** et **description** qui est nettoyée et considérée comme étant la variable explicative textuelle.")
    pickles_apth = "./data/cleaned_data.pkl"
    #if isFileExist(pickles_apth):
    #    print("Reading from pickle file from " + f"{pickles_apth} ...")
    #    df = pd.read_pickle(f"{pickles_apth}")
    #    df_codes = get_codes_df()
    #    df_codes['prdtypecode'] = df_codes['prdtypecode'].astype(int)
    #    df['prdtypecode'] = df['prdtypecode'].astype(int)
    #    df['productid'] = df['productid'].astype('int64')
    #    df['imageid'] = df['imageid'].astype('int64')
    #    df_with_cats = pd.merge(left=df, left_on='prdtypecode', right=df_codes, right_on='prdtypecode' ).sort_values(by='catégorie')
    st.write("**Dimension du dataframe :** " + str(df_with_cats.shape))
    st.write(df_with_cats)
    
    
# TAB TExploration intractive des images  
with tab3:
    col2, col3 = st.columns(2)
    
    
    
    #with col1:
    #    st.html("<h4><span  style='color:orange'>Affichage des images par index (ligne)</span></h4>")
    #    st.write("Le dataset étant brassé, deux index qui se suivent ne donnent pas la même catégorie de produit !")
    #    image_index = st.number_input(
    #        "Sélectionnez un index d'image entre 0 et 82263 et tapez sur ENTREE :", min_value=0, max_value=82263, step =1, value=0, placeholder="tapez un nombre"
    #    )
    #    
    #    if (image_index>=0) and (image_index<82264):
    #        fig = plt.figure(figsize=(7, 7)) 
    #        prod = df_with_cats.iloc[image_index, 0] 
    #        prdid = df_with_cats.iloc[image_index, 2]
    #        imgid = df_with_cats.iloc[image_index, 3]
    #        
    #        #if isFileExist(os.path.join(os.getcwd(), "images/image_train", f"image_{imgid}_product_{prdid}.jpg")):
    #        
    #        # by defaut it's a train image, otherwise open it from test
    #        img = Image.open(requests.get(IMAGES_ROOT +  "/image_train/"  + f"image_{imgid}_product_{prdid}.jpg", stream=True).raw)
    #        if img is None:
    #            img = Image.open(requests.get(IMAGES_ROOT +  "/image_test/"  + f"image_{imgid}_product_{prdid}.jpg", stream=True).raw)
    #            
    #            # img = Image.open(IMAGES_ROOT +  "/image_train/" + f"image_{imgid}_product_{prdid}.jpg")
    #            #img = Image.open(requests.get(IMAGES_ROOT +  "/image_train/"  + f"image_{imgid}_product_{prdid}.jpg", stream=True).raw) 
    #        #else:
    #            #img = Image.open(IMAGES_ROOT +   "/image_test/" + f"image_{imgid}_product_{prdid}.jpg")
    #            #img = Image.open(requests.get(IMAGES_ROOT +  "/image_test/"  + f"image_{imgid}_product_{prdid}.jpg", stream=True).raw)
    #        
    #        # Adds a subplot at the 1st positio 
    #        fig, ax = plt.subplots(1, 1, figsize=(7,7)) 
    #        ax.imshow(img)
    #        ax.set_title(prod)
    #        ax.axis('off') 
    #        st.pyplot(ax.figure)
    #        
    #        st.text("Code de la catégorie :" + str(df_with_cats.iloc[image_index, 4]))
    #        st.text("Catégorie :" + str(df_with_cats.iloc[image_index, 6]))
    
    
    with col2:
        st.html("<h4>⭐<span  style='color:orange'>Affichage aléatoire des images par catégorie sélectionnée</span></h4>")    
        st.write("Cette catégorisation est le résultat d'un travail réalisé à la fois sur des échantillons d'images et du texte.")
        list_elements = []
        for cat in np.sort(df_with_cats['catégorie'].unique()):
            list_elements.append(cat)
        option = st.selectbox(
            "Sélectionnez une catégorie :",
            tuple(list_elements),
        )
        
        #st.text("selected option=" + str(option))
        sel_rows = df_with_cats.loc[df_with_cats['catégorie'] == option]
        idx = np.sort(sel_rows.index)
    
        for i in range(10):
            r = np.random.choice(idx)
            row = df_with_cats.loc[[r]]
            
            # draw 10 randomly selected images
            fig = plt.figure(figsize=(7, 7))
            prod = row['designation']
            prdid = str(int(row['productid']))
            imgid = str(int(row['imageid']))
            code = row['prdtypecode']
            cat = str(row['catégorie'])
            
            #if isFileExist(os.path.join(os.getcwd(), "images/image_train", f"image_{imgid}_product_{prdid}.jpg")):
            #if Image.open(requests.get(IMAGES_ROOT +  "/image_train/"  + f"image_{imgid}_product_{prdid}.jpg", stream=True).raw):    
            #    #img = Image.open(IMAGES_ROOT +  "/image_train/" + f"image_{imgid}_product_{prdid}.jpg")
            #    img = Image.open(requests.get(IMAGES_ROOT +  "/image_train/"  + f"image_{imgid}_product_{prdid}.jpg", stream=True).raw)
            #else:
            #    #img = Image.open(IMAGES_ROOT +  "/image_test/" + f"image_{imgid}_product_{prdid}.jpg")
            #    img = Image.open(requests.get(IMAGES_ROOT +  "/image_test/"  + f"image_{imgid}_product_{prdid}.jpg", stream=True).raw)
            
            img = Image.open(requests.get(IMAGES_ROOT +  "/image_train/"  + f"image_{imgid}_product_{prdid}.jpg", stream=True).raw)
            if img is None:
                img = Image.open(requests.get(IMAGES_ROOT +  "/image_test/"  + f"image_{imgid}_product_{prdid}.jpg", stream=True).raw)
                
            
            # Adds a subplot at the 1st positio 
            fig, ax = plt.subplots(1, 1) 
            ax.imshow(img)
            ax.set_title('Code : ' + str(list(code.values)[0]) + '\n' + prod.values[0])
            ax.axis('off') 
            st.pyplot(ax.figure)
    
    with col3:
        col3.html("<h4>⭐<span  style='color:orange'>Echantillons d'images par code catégorie des produits</span></h4>")
        def drawBtn_2():
            btn_load_images = col3.button("Charger les échantillons",  type="primary" )
            if btn_load_images:
                load_images()
                col3.text("Les images ont été chargées !")
            
        def load_images():
            df_codes = get_codes_df()
            for code, cat in zip(df_codes['prdtypecode'], df_codes['catégorie']):
                col3.write("Catégorie : " + cat)
                col3.image(os.path.join(os.getcwd(), "images", "code-" + str(code) + ".png"))
                #st.image(Image.open(requests.get(IMAGES_ROOT + "/code-" + str(code) + ".png", stream=True).raw) )
            
        drawBtn_2()
    