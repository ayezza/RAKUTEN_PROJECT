import os
import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import requests


# This is relative root directory af trai and test images
#IMAGES_ROOT = os.path.join(os.getcwd(), "images")
IMAGES_ROOT = r"https://www.anigraphics.fr/images"


st.title("Résultats obtenus")


tabs_title = ["🗃Texte uniquement", "🗃Image uniquement", "🗃🗃Bimodal texte & image"]
tab1, tab2, tab3 = st.tabs(tabs_title)

# TAB partie texte
with tab1: 
    
    st.write("### Modèles Machine Learning (ML)")
    col1, col2, col3= st.columns(3)
    col1.metric("SVM", "Accuracy", "77%", "off")
    col1.metric("SVM", "f1_score", "77%", "off")
    col1.write("Le **SVM** prône l'overfitting et n'a pas été retenu.")
    col2.metric("Logistic Regression ", "Accuracy", "80%", "normal")
    col2.metric("Logistic Regression ", "f1_score", "79%", "normal")
    col2.write("Le **Logistic Regression** fournit de très bonnes performances.")
    #img_lr = Image.open(IMAGES_ROOT + "/log-reg-lear_curve.png")
    img_lr = Image.open(requests.get(IMAGES_ROOT +  "/"  + "log-reg-lear_curve.png", stream=True).raw)
    col3.text("Logistic Regression-Courbes des apprentissages")
    col3.image(img_lr)
    #img_lr = Image.open(IMAGES_ROOT + "/log-reg-cm.png")
    img_lr = Image.open(requests.get(IMAGES_ROOT +  "/"  + "llog-reg-cm.png", stream=True).raw)
    col3.text("Logistic Regression-Matrice de Confusion")
    col3.image(img_lr)
    
    st.html("<hr>")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("KNN", "Accuracy", "55%", "off")
    col1.metric("KNN", "f1_score", "55%", "off")
    col1.write("Le **KNN** ne fournit pas des résultats satisfaisants.")
    col2.metric("XGBOOST", "Accuracy", "78%", "normal")
    col2.metric("XGBOOST", "f1_score", "78%", "normal")
    col2.write("Le **XGBOOST** fournit des bons résultats presque au même niveau que le Logistic regession.")
    #img_lr = Image.open(IMAGES_ROOT + "/xgboost-lear_curve.png")
    img_lr = Image.open(requests.get(IMAGES_ROOT +  "/"  + "xgboost-lear_curve.png", stream=True).raw)
    col3.text("XGBOOST-Courbes des apprentissages")
    col3.image(img_lr)
    

    st.html("<hr>")
     
    st.write("### Modèles Réseaux de Neurones")
    col1, col2 = st.columns(2)
    col1.metric("LSTM combiné avec Conv1D", "Accuracy", "79%", "normal")
    col1.metric("LSTM combiné avec Conv1D", "f1_score", "79%", "normal")
    #img = Image.open(IMAGES_ROOT + "/lstm-evaluation.png")
    img = Image.open(requests.get(IMAGES_ROOT +  "/"  + "lstm-evaluation.png", stream=True).raw)
    col1.text("Evaluation du modèle sur l'ensemble de test")
    col1.image(img)
    #img = Image.open(IMAGES_ROOT + "/lstm-conv1d-accu_loss.png")
    img = Image.open(requests.get(IMAGES_ROOT +  "/"  + "lstm-conv1d-accu_loss.png", stream=True).raw)
    col2.text("Evolution de l'accuracy et de la perte")
    col2.image(img)
    #img = Image.open(IMAGES_ROOT + "/lstm-conv1d-cm.png")
    img = Image.open(requests.get(IMAGES_ROOT +  "/"  + "lstm-conv1d-cm.png", stream=True).raw)
    col2.text("LSTM combiné avec Conv1D-Matrice de confusion")
    col2.image(img)
    
    st.html("<hr>")
    
    col1, col2 = st.columns(2)
    col1.metric("LSTM renforcé avec GLOVE", "Accuracy", "80%", "normal")
    col1.metric("LSTM renforcé avec GLOVE", "f1_score", "80%", "normal")
    col1.write("GLOVE alié à LSTM semble améliorer les performances. \
        Le fait qu'il comprend plus de mots en anglais qu'en français n'a pa eu l'ffect escompté.")
    #img = Image.open(IMAGES_ROOT + "/lstm-glove-accu_loss.png")
    img = Image.open(requests.get(IMAGES_ROOT +  "/"  + "lstm-glove-accu_loss.png", stream=True).raw)
    col2.text("Evolution de l'accuracy et de la perte")
    col2.image(img)
    #img = Image.open(IMAGES_ROOT + "/lstm-glove-cm.png")
    img = Image.open(requests.get(IMAGES_ROOT +  "/"  + "lstm-glove-cm.png", stream=True).raw)
    col2.text("LSTM combiné avec Conv1D - Matrice de confusion")
    col2.image(img)
    
    st.html("<hr>")
    
    col1, col2 = st.columns(2)
    col1.metric("CNN (Conv1D)", "Accuracy", "80%", "normal")  
    col1.metric("CNN (Conv1D)", "f1_score", "80%", "normal")  
    #img = Image.open(IMAGES_ROOT + "/conv1d-accu_loss.png")
    img = Image.open(requests.get(IMAGES_ROOT +  "/"  + "conv1d-accu_loss", stream=True).raw)
    col2.text("Evolution de l'accuracy et de la perte")
    col2.image(img)
    #img = Image.open(IMAGES_ROOT + "/conv1d-mc.png")
    img = Image.open(requests.get(IMAGES_ROOT +  "/"  + "conv1d-mc", stream=True).raw)
    col2.text("Conv1D - Matrice de confusion")
    col2.image(img)
    
    st.html("<span style='color:green;text-align:center;font-size:24px;height:32px;background-color:#FFFFFF;'>\
        En résumé, Les deux typologies des modèles ML et RNN sélectionnés fournissent le même niveau de performances.</span>")
    
    
    st.html("<hr>")
    
    
    st.write("### Modèles ML (approche différente)")
    st.write("Les mots ne sont pas vectorisés, mais transformés en variables descriptive avec un nombre limité à 300.\
    Cette approche n'a pas été retenue pour des questions de performance d'évaluation des ptrédictions.")
    img_approche_ml = Image.open(os.path.join(os.getcwd(), "images", "image-2.png"))
    st.image(img_approche_ml)
    
    st.html("<hr>")
    

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("KNN", "accuracy", "90%", "normal")
    col2.metric("KNN avec une réduction PCA de 57% des variables", "accuracy", "87%", "normal")
    col3.metric("Random Forest", "accuracy", "91%", "normal")
    col4.metric("Logistic Regression", "accuracy", "89%", "normal")
    col5.metric("SVC (Support Vector)", "accuracy", "85%", "normal")
    
    
with tab2:
    st.header("Résultats obtenus")

     
    st.write("### Modèles **baseline** images")
    col1, col2 = st.columns(2)
    col1.metric("PCA et Random Forest", "Accuracy", "+49%", delta_color= "inverse")
    col1.metric("PCA et Random Forest", "f1_score (moy)", "+56%", delta_color= "inverse")
    col1.text("Rapport de classification")
    #img = Image.open(IMAGES_ROOT + "/pca+RF_classif_report.png")
    img = Image.open(requests.get(IMAGES_ROOT +  "/"  + "pca+RF_classif_report", stream=True).raw)
    col1.image(img)
    #img = Image.open(IMAGES_ROOT + "/PCA+RF_confusion_matrix.png")
    img = Image.open(requests.get(IMAGES_ROOT +  "/"  + "PCA+RF_confusion_matrix", stream=True).raw)
    col2.text("Matrice de confusion")
    col2.image(img)
    st.write(">La réduction **PCA** appliquée aux images nous a permis de conserver **90%** de la variance expliquée des images en \
            réduisant leur taille de **4096 features (64x64) à 278**.")
    st.write(">Au-delà de 200 estimateurs l'accuracy ne progresse plus")
    st.write(">Des écarts relativement importants en termes d'accuracy des 27 classes, \
        ce qui se répercute directement sur la matrice de confusion illustrée ci-dessus.")
    st.write("Le **XGBOOST couplé avec une PCA** fournit les mêmes performances que Random Forest avec un temps d'exécution plus long.")
    st.write("En conclusion, la PCA enlève trop d'information des données image pour obtenir un résultat optimal.")
    
    st.html("<hr>")
    
    st.write("### Modèles **DEEP-LEARNING** images")
    col1, col2 = st.columns(2)
    col1.metric("RESNET 50", "Accuracy", "+48%", delta_color= "inverse")
    col1.write(">Le **RESTNET 50** s'est arrêté de progresser au bout de 36 EOCHS à 48% d'accuracy")
    col2.text("RESNET 50 - Tendances de l'accuracy et de la perte")
    #img = Image.open(IMAGES_ROOT + "/acc_loss_resnet.png")
    img = Image.open(requests.get(IMAGES_ROOT +  "/"  + "acc_loss_resnet", stream=True).raw)
    col2.image(img)
    
    st.html("<hr>")
    
    
    col1, col2 = st.columns(2)
    col1.metric("EfficentNet B5", "Accuracy", "+46%", delta_color= "inverse")
    col1.write(">Le **EfficentNet B5** s'est arrêté de progresser au bout de 15 EOCHS à 46% d'accuracy")
    col2.text("EfficentNet B5 - Tendances de l'accuracy et de la perte")
    #img = Image.open(IMAGES_ROOT + "/acc_loss_effnet.png")
    img = Image.open(requests.get(IMAGES_ROOT +  "/"  + "acc_loss_effnet", stream=True).raw)
    col2.image(img)
    
    st.html("<hr>")
    
    col1, col2 = st.columns(2)
    col1.metric("ViT (Vision Transformer)", "Accuracy", "+52%", delta_color= "inverse")
    col1.write(">Le **ViT** a bien terminé ses 10 EPOCHs avec 52% d'accuracy")
    col2.text("ViT - Tendances de l'accuracy et de la perte")
    #img = Image.open(IMAGES_ROOT + "/acc_loss_vit.png")
    img = Image.open(requests.get(IMAGES_ROOT +  "/"  + "acc_loss_vit", stream=True).raw)
    col2.image(img)
    
    st.write(">Les trois modèle offrent un niveau de précision presque équivalent à celui des modèles baseline")
    st.write(">les frontières entre certaines catégories est assez mince pour entraîner une confusion des modèles \
        dans leurs prédictions, comme par exemple 'consoles de jeu', 'consoles, jeux & équipement d'occasion'")
    


with tab3:
    st.header("Résultats obtenus")
    