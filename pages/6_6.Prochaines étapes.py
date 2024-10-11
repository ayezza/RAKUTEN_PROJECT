import streamlit as st

st.title("Prochaines étape")


st.write("Bien que les résultats obtenus soient d’un très bon niveau, voire excellents pour le modèle bimodal, il y aura toujours \
    une fenêtre d’amélioration en appliquant les dernières avancées en matière de modélisation. Plus précisément, les aspects suivants \
    sont assujettis à un travail plus fin et plus poussé :")
st.write(">1. Procéder à rendre le dataset plus équilibré avec une démarche plus approfondie au travers l’enrichissement par des données actuelles qu’elles soient du texte ou des images.")
st.write(">2.	Écarter le déséquilibre de notre jeu de données en appliquant d’autres approches d’ensembling plus performantes.") 
st.write(">3.	Effectuer un preprocessing des images qui présentent un fond blanc important afin de faire apparaître les contours.")
st.write(">4.	Une fois disposé des capacités de calcul suffisantes, notamment pour le traitement des images et l’utilisation des \
    modèles pré-entrainés comme BERT, reprendre cette partie afin d’améliorer encore plus les résultats obtenus.")
st.write(">4. Finalemnt, s'inspirer des modèles élaborés afin des les appliquer sur d'autres problèmes de même nature ou se rapprochant du problème présent.")
