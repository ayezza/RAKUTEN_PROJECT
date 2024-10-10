from pathlib import Path
import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


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

'''
import io
buffer = io.StringIO()
df_with_cats.info(buf=buffer)
s = buffer.getvalue()
st.text(s)
st.dataframe(df_with_cats)
'''

# Configuration des tabs
tabs_title = ["🗃Partie texte", "🗃Partie images"]
tab1, tab2 = st.tabs(tabs_title)


# TAB partie texte
with tab1:
    img_analyse_graphe_1 = Image.open(os.path.join(os.getcwd(), "images", "analyse_graphe_1.png"))
    img_analyse_graphe_2 = Image.open(os.path.join(os.getcwd(), "images", "analyse_graphe_2.png"))
    img_analyse_graphe_3 = Image.open(os.path.join(os.getcwd(), "images", "analyse_graphe_3.png"))
    img_analyse_graphe_4 = Image.open(os.path.join(os.getcwd(), "images", "analyse_graphe_4.png"))
    
    st.header("Graphes de répartitions avant le cleanning")
    
    st.html("<h3><span  style='color:orange'>Répartition des catégories selon les decriptions associées :</span></h3>")
    st.image(img_analyse_graphe_1)
    
    st.html("<h3><span  style='color:orange'>Répartition des catégories selon les designations associées :</span></h3>")
    st.image(img_analyse_graphe_2)
    
    st.header("Répartition des catégories selon les designations associées et cartographie en cloud des mots après le cleaning")
    st.html("Les actions suivantes ont été menées pour donner la nouvelle répartition des catégories ci-dessous un peu mieux équilibrée que précédemment\
           <ul><li>Suppression des valeurs nulles</li><ul>\
            <ul><li>Suppression des doublons</li><ul>\
            <ul><li>Suppression des expressions n'apportant aucune valeur sémantique relative au produit</li><ul>\
            <ul><li>Ajout d'une variable descriptive des 27 catgories à partir de l'analyse des images associées aux produits</li><ul>")
    st.image(img_analyse_graphe_3)
    
    st.html("<p>Les mots en fonction de leur taille dans le cloud, révèlent leurs fréquence dans les deux variables explicatives combinées,\
        la désignation des produits et la description associée. D'une manière indirecte, les mots mis plus en avant révèlent la catégorie des produits\
            la plus dominante en terme de description comme les mot <span style='color: red'><strong>jeu, enfant, sac, piscine...</strong></span></p>")
    st.image(img_analyse_graphe_4)
    
    
    st.header("Graphes de répartitions après le cleanning en live")
    #   Load some graphs in live
    def drawBtn_1():
        btn_load_graphs = st.button("Charger les graphes",  type="primary" )
        if btn_load_graphs:
            load_graphs()
            st.text("Les graphes ont été chargés !")
    
    def load_graphs():  
        if isFileExist(pickles_apth):
            sns.set_theme(rc={'figure.figsize': (10, 7)})
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
            
            g3 =sns.countplot(data=df_with_cats,  x='prdtypecode',  orient='v', palette='Spectral')
            g3.set_title("Distribution en nombre de produits par catégorie")
            g3.set_xlabel('Catégories')
            g3.set_ylabel('Nombre de produits')
            ax.tick_params(axis='x', rotation=90)
            plt.legend(loc='lower left')
            #g3.set_xticklabels(labels=df_with_cats["catégorie"].unique())
            st.pyplot(g3.figure)
            
            #st.bar_chart(df_with_cats, x="catégorie", y="prdtypecode", color="catégorie", x_label ="Catégories", y_label="Nombre de produits", stack=False)
            
        else:
            st.html("Fichier PICKLE introuvable ici : " + pickles_apth)

    # action button
    drawBtn_1()
    
# TAB partie images    
with tab2:
    st.html("<h4><span  style='color:orange'>Echantillons d'images par code catégorie des produits</span></h4>")
    def drawBtn_2():
        btn_load_images = st.button("Charger les échantillons",  type="primary" )
        if btn_load_images:
            load_images()
            st.text("Les images ont été chargées !")
        
    def load_images():
        df_codes = get_codes_df()
        for code, cat in zip(df_codes['prdtypecode'], df_codes['catégorie']):
            st.write("Catégorie : " + cat)
            st.image(Image.open(os.path.join(os.getcwd(), "images", "code-" + str(code) + ".png")))
            
        
    drawBtn_2()
    
    
    st.html("<hr")
    st.html("<h4><span  style='color:orange'>Affichage des images par index</span></h4>")
    image_index = st.number_input(
        "Sélectionnez un index d'image entre 0 et 82264 et tapez sur ENTREE :", min_value=0, max_value=82264, step =1, value=0, placeholder="tapez un nombre"
    )
    
    if (image_index>=0) and (image_index<82264):
        fig = plt.figure(figsize=(7, 7)) 
        prod = df_with_cats.iloc[image_index, 0] 
        prdid = df_with_cats.iloc[image_index, 2]
        imgid = df_with_cats.iloc[image_index, 3]
        
        if isFileExist(os.path.join(os.getcwd(), "images/image_train", f"image_{imgid}_product_{prdid}.jpg")):
            img = Image.open(os.path.join(os.getcwd(), "images/image_train", f"image_{imgid}_product_{prdid}.jpg"))
        else:
            img = Image.open(os.path.join(os.getcwd(), "images/image_test", f"image_{imgid}_product_{prdid}.jpg"))
        
        # Adds a subplot at the 1st positio 
        fig, ax = plt.subplots(1, 1, figsize=(7,7)) 
        ax.imshow(img)
        ax.set_title(prod)
        ax.axis('off') 
        st.pyplot(ax.figure)
    
    
    
    
    
    st.html("<h4><span  style='color:orange'>Affichage des images par catégorie sélectionnée</span></h4>")    
    list_elements = []
    for cat in np.sort(df_with_cats['catégorie'].unique()):
        list_elements.append(cat)
    option = st.selectbox(
        "Sélectionnez une catégorie :",
        tuple(list_elements),
    )
    
    #st.text("selected option=" + str(option))
    sel_rows = df_with_cats.loc[df_with_cats['catégorie'] == option]
    #st.text("liste of selected rows :\n" +  str(sel_rows))
    idx = np.sort(sel_rows.index)
    #st.text("index=")
    #st.text(idx)
   
    for i in range(10):
        r = np.random.choice(idx)
        #st.text('r = ' + str(r))
        
        row = df_with_cats.loc[[r]]
        #st.text(row)
        #st.text(row.index)
        #
        #st.text('designation : ' + row['designation'])
        #st.text('description :' + row['description'])
        #st.text('code : ' + str(row['prdtypecode']))
        #st.text('catégorie : ' + str(row['catégorie']))
        
        # draw 10 randomly selected images
        fig = plt.figure(figsize=(7, 7))
        prod = row['designation']
        prdid = str(int(row['productid']))
        imgid = str(int(row['imageid']))
        cat = str(row['catégorie'])
        #st.text(prod)
        #st.text(prdid)
        #st.text(imgid)
        if isFileExist(os.path.join(os.getcwd(), "images/image_train", f"image_{imgid}_product_{prdid}.jpg")):
            img = Image.open(os.path.join(os.getcwd(), "images/image_train", f"image_{imgid}_product_{prdid}.jpg"))
        else:
            img = Image.open(os.path.join(os.getcwd(), "images/image_test", f"image_{imgid}_product_{prdid}.jpg"))
        
        # Adds a subplot at the 1st positio 
        fig, ax = plt.subplots(1, 1) 
        ax.imshow(img)
        ax.set_title(prod)
        ax.axis('off') 
        st.pyplot(ax.figure)
        
    