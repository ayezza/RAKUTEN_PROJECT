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


tabs_title = ["🗃Texte", "🗃Image", "🗃🗃Texte & image"]
tab1, tab2, tab3 = st.tabs(tabs_title)

# TAB partie texte
with tab1: 
    st.write("### **Méthode d'évaluation :**")
    st.write("1. Factorisation du pre-proc & vectorisation avec les pipeline")
    st.write("2. Cross-validation avec un StratifiedKFold de 5 splits")
    st.write("3. Choix des métriques : balanced accuracy, f1 score, roc-auc et geo (Classes désiquilibrées)")
    st.divider()
    st.write("#### **6 modèles ML, 4 Deep :** Pour chaque modèle")
    st.write("- Rapport de classification")
    st.write("- Matrice de confusion")
    st.write("- Graphe roc-au")
    st.write("- Learning curve")
    st.divider()
    st.write("- Sélection du best modèle")
    st.write("- Optimisation du best modèle avec un **GridSearchCV**")
    st.divider()
    
    
    st.write("### Modèles Machine Learning (ML)")
    col1, col2, col3= st.columns(3)
    col1.metric("SVM", "Accuracy", "77%", "off")
    col1.metric("SVM", "f1_score", "77%", "off")
    col1.write("Le **SVM** prône l'overfitting et n'a pas été retenu.")
    col2.metric("Logistic Regression ", "Accuracy", "80%", "normal")
    col2.metric("Logistic Regression ", "f1_score", "79%", "normal")
    col2.write("Le **Logistic Regression** fournit de très bonnes performances.")
    img_lr = Image.open(os.path.join(os.getcwd(), "images", "log-reg-lear_curve.png"))
    #img_lr = Image.open(requests.get(IMAGES_ROOT +  "/"  + "log-reg-lear_curve.png", stream=True).raw)
    col3.text("Logistic Regression-Courbes des apprentissages")
    col3.image(img_lr)
    img_lr = Image.open(os.path.join(os.getcwd(), "images", "log-reg-cm.png"))
    #img_lr = Image.open(requests.get(IMAGES_ROOT +  "/"  + "log-reg-cm.png", stream=True).raw)
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
    #img_lr = Image.open(requests.get(IMAGES_ROOT +  "/"  + "xgboost-lear_curve.png", stream=True).raw)
    img_lr = Image.open(os.path.join(os.getcwd(), "images", "xgboost-lear_curve.png"))
    col3.text("XGBOOST-Courbes des apprentissages")
    col3.image(img_lr)
    

    st.html("<hr>")
     
    st.write("### Modèles Réseaux de Neurones")
    st.write("BERT (30% data)")
    col1, col2 = st.columns(2)
    col1.metric("BERT", "Accuracy", "79%", "normal")
    col1.metric("BERT", "f1_score", "78%", "normal")
    st.divider()
    col1, col2 = st.columns(2)
    col1.metric("LSTM combiné avec Conv1D", "Accuracy", "79%", "normal")
    col1.metric("LSTM combiné avec Conv1D", "f1_score", "79%", "normal")
    #img = Image.open(IMAGES_ROOT + "/lstm-evaluation.png")
    #img = Image.open(requests.get(IMAGES_ROOT +  "/"  + "lstm-evaluation.png", stream=True).raw)
    img = Image.open(os.path.join(os.getcwd(), "images", "lstm-evaluation.png"))
    col1.text("Evaluation du modèle sur l'ensemble de test")
    col1.image(img)
    #img = Image.open(IMAGES_ROOT + "/lstm-conv1d-accu_loss.png")
    #img = Image.open(requests.get(IMAGES_ROOT +  "/"  + "lstm-conv1d-accu_loss.png", stream=True).raw)
    img = Image.open(os.path.join(os.getcwd(), "images", "lstm-conv1d-accu_loss.png"))
    col2.text("Evolution de l'accuracy et de la perte")
    col2.image(img)
    #img = Image.open(IMAGES_ROOT + "/lstm-conv1d-cm.png")
    #img = Image.open(requests.get(IMAGES_ROOT +  "/"  + "lstm-conv1d-cm.png", stream=True).raw)
    img = Image.open(os.path.join(os.getcwd(), "images", "lstm-conv1d-cm.png"))
    col2.text("LSTM combiné avec Conv1D-Matrice de confusion")
    col2.image(img)
    
    st.html("<hr>")
    
    col1, col2 = st.columns(2)
    col1.metric("LSTM renforcé avec GLOVE", "Accuracy", "80%", "normal")
    col1.metric("LSTM renforcé avec GLOVE", "f1_score", "80%", "normal")
    col1.write("GLOVE alié à LSTM semble améliorer les performances. \
        Le fait qu'il comprend plus de mots en anglais qu'en français n'a pa eu l'ffect escompté.")
    #img = Image.open(IMAGES_ROOT + "/lstm-glove-accu_loss.png")
    #img = Image.open(requests.get(IMAGES_ROOT +  "/"  + "lstm-glove-accu_loss.png", stream=True).raw)
    img = Image.open(os.path.join(os.getcwd(), "images", "lstm-glove-accu_loss.png"))
    col2.text("Evolution de l'accuracy et de la perte")
    col2.image(img)
    #img = Image.open(IMAGES_ROOT + "/lstm-glove-cm.png")
    #img = Image.open(requests.get(IMAGES_ROOT +  "/"  + "lstm-glove-cm.png", stream=True).raw)
    img = Image.open(os.path.join(os.getcwd(), "images", "lstm-glove-cm.png"))
    col2.text("LSTM combiné avec Conv1D - Matrice de confusion")
    col2.image(img)
    
    st.html("<hr>")
    
    col1, col2 = st.columns(2)
    col1.metric("CNN (Conv1D)", "Accuracy", "80%", "normal")  
    col1.metric("CNN (Conv1D)", "f1_score", "80%", "normal")  
    #img = Image.open(IMAGES_ROOT + "/conv1d-accu_loss.png")
    #img = Image.open(requests.get(IMAGES_ROOT +  "/"  + "conv1d-accu_loss", stream=True).raw)
    img = Image.open(os.path.join(os.getcwd(), "images", "conv1d-accu_loss.png"))
    col2.text("Evolution de l'accuracy et de la perte")
    col2.image(img)
    #img = Image.open(IMAGES_ROOT + "/conv1d-mc.png")
    #img = Image.open(requests.get(IMAGES_ROOT +  "/"  + "conv1d-mc.png", stream=True).raw)
    img = Image.open(os.path.join(os.getcwd(), "images", "conv1d-mc.png"))
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
    #img = Image.open(requests.get(IMAGES_ROOT +  "/"  + "PCA+RF_confusion_matrix.png", stream=True).raw)
    img = Image.open(os.path.join(os.getcwd(), "images", "PCA+RF_confusion_matrix.png"))
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
    #img = Image.open(requests.get(IMAGES_ROOT +  "/"  + "acc_loss_resnet", stream=True).raw)
    img = Image.open(os.path.join(os.getcwd(), "images", "acc_loss_resnet.png"))
    col2.image(img)
    
    st.html("<hr>")
    
    
    col1, col2 = st.columns(2)
    col1.metric("EfficentNet B5", "Accuracy", "+46%", delta_color= "inverse")
    col1.write(">Le **EfficentNet B5** s'est arrêté de progresser au bout de 15 EOCHS à 46% d'accuracy")
    col2.text("EfficentNet B5 - Tendances de l'accuracy et de la perte")
    #img = Image.open(IMAGES_ROOT + "/acc_loss_effnet.png")
    #img = Image.open(requests.get(IMAGES_ROOT +  "/"  + "acc_loss_effnet", stream=True).raw)
    img = Image.open(os.path.join(os.getcwd(), "images", "acc_loss_effnet.png"))
    col2.image(img)
    
    st.html("<hr>")
    
    col1, col2 = st.columns(2)
    col1.metric("ViT (Vision Transformer)", "Accuracy", "+52%", delta_color= "inverse")
    col1.write(">Le **ViT** a bien terminé ses 10 EPOCHs avec 52% d'accuracy")
    col2.text("ViT - Tendances de l'accuracy et de la perte")
    #img = Image.open(IMAGES_ROOT + "/acc_loss_vit.png")
    #img = Image.open(requests.get(IMAGES_ROOT +  "/"  + "acc_loss_vit", stream=True).raw)
    img = Image.open(os.path.join(os.getcwd(), "images", "acc_loss_vit.png"))
    col2.image(img)
    
    st.write(">Les trois modèle offrent un niveau de précision presque équivalent à celui des modèles baseline")
    st.write(">les frontières entre certaines catégories est assez mince pour entraîner une confusion des modèles \
        dans leurs prédictions, comme par exemple 'consoles de jeu', 'consoles, jeux & équipement d'occasion'")
    


with tab3:
    st.header("Trois approches différentes ont été adoptées :")
    
    col1, col2 = st.columns(2)
    col1.write("1. #### **Entraîner un modèle multimodal :**")
    col1.write(">Nous avons entraîné le **CLIP (Contrastive Language-Image pretraining) d'OPEN AI**, qui associe des paires de mots/images dans un espace \
     vectoriel et apprend à les différencier en les rapprochant ou en les éloignant. \
     Malheureusement, la durée d'entraînement étant trop longue avec des ressources trop limitées, le modèle a été arrêté au bout de 5 époques.")
    
    img = Image.open(os.path.join(os.getcwd(), "images", "acc_loss_clip.jpg"))
    col2.image(img)
    
    
    st.divider()
    col1, col2 = st.columns(2)
    col1.write("2. #### **Entraîner un modèle de concaténation text a images :**")
    col1.write(">**Le principe :** prendre les meilleurs modèles texte + meilleur modèle image + couche de classification")
    col1.write(">Si on compare les rapports de classification/matrice de confusion des meilleurs modèles texte/image, on se rend compte que les modèles \
     peuvent se compenser (certaines classes sont bien catégorisées par le modèle image et moins bien par celui du texte, et inversement")
    
    img = Image.open(os.path.join(os.getcwd(), "images", "comp_matrix_txt_img.jpg"))
    col2.image(img)
    
    
    st.divider()
    col1, col2 = st.columns(2)
    col1.write("3. #### **Entraîner un modèle hybride :**")
    col1.write("On entraîne donc un modèle qui prend en entrée texte + image. \
        Le texte est passé dans la LR texte gelée et produit un vecteur de probas de 27 classes. \
        L'image est passée dans le ViT gelé et produit un vecteur de probabilités des 27 classes également. \
        On ajoute 3 couches en sortie : 2 classif + 1 dropout pour éviter l'overfitting. \
        Elles prennent 54 entrées (les probabilités des 2 modèles) et produisent 27 sorties (les catégories). \
        Les features entraînables sont les probabilités de sortie des 2 modèles. \
        Le résultat  : **Accuracy de près de 95%** sur la validation dès la 1ère EPOCH. \
        On a donc arrêté le modèle car l'entraînementest  très long et performe très bien.")
   