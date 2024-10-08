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


#import io
#buffer = io.StringIO()
#df_with_cats.info(buf=buffer)
#s = buffer.getvalue()
#st.text(s)
#st.dataframe(df_with_cats)


# Configuration des tabs
tabs_title = ["🗃Texte uniquement", "🗃Images uniquement", "🗃Exploration intéractive des images"]
tab1, tab2, tab3 = st.tabs(tabs_title)


# TAB Analyse du texte
with tab1:
    img_analyse_graphe_1 = Image.open(os.path.join(os.getcwd(), "images", "analyse_graphe_1.png"))
    img_analyse_graphe_2 = Image.open(os.path.join(os.getcwd(), "images", "analyse_graphe_2.png"))
    img_analyse_graphe_3 = Image.open(os.path.join(os.getcwd(), "images", "analyse_graphe_3.png"))
    img_analyse_graphe_4 = Image.open(os.path.join(os.getcwd(), "images", "analyse_graphe_4.png"))
    
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
        st.write("> H0 : la taille des images n'a pas d'influence sur la catégorie")
        st.write("> H1 : la taille des images a une influence sur la catégorie")
        st.write("Au vu des résulats (**p-value très inférieure à 0.05**) et **F-stat très élevée (+ de 339)**, on peut rejeter \
                l'hypothèse H0 au profit de la H1 : la taille des images a une influence sur la catégorie. \
                C'est donc une feature qui pourrait éventuellement nous servir par la suite pour catégoriser les images, \
                mais nous n'en aurons pas besoin au final car les features les plus évidentes (valeurs de pixels des img) suffiront.")

    st.html("<hr")


    img_analyse_images_1 = Image.open(os.path.join(os.getcwd(), "images", "exemple_preprocess_baseline.png"))
    img_analyse_images_2 = Image.open(os.path.join(os.getcwd(), "images", "exemple_preprocess_deeplearning.png"))
    img_analyse_images_3 = Image.open(os.path.join(os.getcwd(), "images", "exemple_preprocess_generique.png"))
    
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
   

    
    st.write("Nous avons effectué 3 types de preprocessing:")
    
    st.write("-> **Un preprocessing générique**: utilisable par tout type de modèles. Il contient 1000 images par catégorie, et \
    30% des images ont été **augmentées** (rotations, zoom, etc.) pour diversifier le dataset. Leur taille est inchangée  \
    (500x500) et pourra par la suite être adaptée en fonction des modèles (voir **exemple_preprocess_generique**")
    
    st.write("-> un preprocessing pour les modèles de Deep Learning : les modèles de Deep Learning sont inclues dans des  \
    librairies qui contiennent leurs propres fonctions de preprocessing (voici un exemple de dataset image  \
    preprocessé par keras de Tensorflow: **exemple_preprocess_deeplearning**). ce preprocessing sera adapté au cas par cas en fonction  \
    des modèles.")
    
    st.write("-> un preprocessing pour les modèles **baseline** : Nous allons entraîner par la suite 2 types de modèles : des \
    modèles deep learning (plus complexes) et des modèles baseline (plus simples).  \
    Pour les modèles baseline, nous avons réduit les images en 64x64 pixels (au lieu de 500x500). \
    Nous les avons passées en niveaux de gris. (voir graphe **exemple_preprocess_baseline**")


    
    
# TAB TExploration intractive des images  
with tab3:
    col1, col2 = st.columns(2)
    
    with col1:
        st.html("<h4><span  style='color:orange'>Affichage des images par index (ligne)</span></h4>")
        st.write("Le dataset étant brassé, deux index qui se suivent ne donnent pas la même catégorie de produit !")
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
    
    with col2:
        st.html("<h4><span  style='color:orange'>Affichage des images par catégorie sélectionnée</span></h4>")    
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
            cat = str(row['catégorie'])
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
        
    
    st.html("<hr>")
    
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
    