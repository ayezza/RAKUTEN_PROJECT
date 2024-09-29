from pathlib import Path
import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import os



st.title("Jeu de données")

# Configuration de la barre latérale
tabs_title = ["🗃Jeu de données intial", "🗃Jeu de données cible"]
tab1, tab2 = st.tabs(tabs_title)
cur_dir = os.getcwd()
st.write(cur_dir)

with tab1:
    img_train_set = Image.open(os.path.join(os.getcwd(), "images", "train_set.png"))
    img_test_set = Image.open(os.path.join(os.getcwd(), "images", "test_set.png"))
    img_target_set = Image.open(os.path.join(os.getcwd(), "images", "target_set.png"))
    img_schema_dataset = Image.open(os.path.join(os.getcwd(), "images", "schema_source_dataset.png"))
    
    st.html("<h1><span  style='color:orange'>Schéma des jeux de données d'entraînement et de test :</span></h1>")
    st.image(img_schema_dataset)
    
    st.html("<h1><span  style='color:orange'>Jeu de données d'entraînement :</span></h1>")
    st.image(image=img_train_set)
    
    st.html("<h1><span  style='color:orange'>Jeu de données de test :</spane></h1>")
    st.image(image=img_test_set)
    
    st.html("<h1><span  style='color:orange'>Remarque :</span></h1>")
    st.html("<p>Aussi bien le jeux de données d'entraînement que celui de test,\
        ne contienent les données relatives à la cible, i.e., les catégories associées aux produits.</p>\
        Le seul jeu de données contenant la cible, correspond uniquement au jeu de données d'entrainement ci-dessous (84 916 lignes)")
    
    img_shema_target = Image.open(os.path.join(os.getcwd(), "images", "shema_target.png"))
    st.html("<h1><span  style='color:orange'>Schéma de la variable cible :</span></h1>")
    st.html("<p>La variable cible est formée de 27 catégories numériques. Une variable traduisant les catégories numériques \
        ci-dessous en catégories descirptives sur la base de la combinaison du texte et des images associées aux produits.</p>")
    st.image(img_shema_target)
    st.html("<h2><span  style='color:orange'>List des catégories descriptives :</span></h2>")
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
    st.dataframe(df)
    
    st.html("<h1><span  style='color:orange'>Jeu de données cible pour la partie entraînement</span></h1>")
    st.image(img_target_set)
    
with tab2:
    img_datset_cibe = Image.open(os.path.join(os.getcwd(), "images", "datset_cibe.png"))
    st.html("<h1><span  style='color:orange'>Schéma du jeu de données cible construit :</span></h1>")
    st.image(img_datset_cibe)
    
    st.html("<p>Les deux variables <strong>'designation'</strong> et <strong>'description'</strong> ont été concténées \
        dans une une unique variable <strong>'desi_desc'</strong>.\
        Le reste des variable explicatives ont été préservées.</p>")
   
    '''
    print("Reading from pickle file...")
    st.write(f"../../../../data/cleaned_data.pkl")
    df = pd.read_pickle(f"../../../../data/cleaned_data.pkl")

    # ajouter la colone nom_image indiquant le nom du fichier image
    df['imageid']= df['imageid']
    df['prdtypecode']= df['prdtypecode']
    df['nom_image'] = "image" + "_" + df['imageid'].astype('str') + "_product_" + df['prdtypecode'].astype('str') + ".jpg"
    unique_prod_codes = np.sort(df['prdtypecode'].unique())
    df['prdtypecode_encoded'] =  df['prdtypecode'].apply(lambda x: np.where(unique_prod_codes==x)[0][0])
    print(df.head())
    '''