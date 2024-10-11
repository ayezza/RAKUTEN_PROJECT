from pathlib import Path
import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import torch
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
import joblib
import torch.nn as nn
from sklearn.model_selection import train_test_split
import os
st.set_page_config(layout="wide")

df = pd.read_csv("/Users/ouissamgouni/Documents/it-workspace/bootcamp-mle-24/git/final/jul24_bds_rakuten/data/csv_files/img-text-clean-data.csv")
DATA_DIR="/Users/ouissamgouni/Documents/it-workspace/bootcamp-mle-24/git/final/jul24_bds_rakuten/data/processed/img_classified_by_prdtypecode"
NUM_CLASSES = 27

prdtypecode_list = ['10', '1140', '1160', '1180', '1280', '1281', '1300', '1301', 
                    '1302', '1320', '1560', '1920', '1940', '2060', '2220', 
                    '2280', '2403', '2462', '2522', '2582', '2583', '2585', 
                    '2705', '2905', '40', '50', '60']

# Tri de la liste par ordre ascendant
sorted_prdtypecode_list = sorted(prdtypecode_list, key = int)

# Dict avec "key":"value" = "prdtypecode":"index" (ou "index" est l'index du "prdtypecode dans la liste triée")
prdtypecode_dict = {code: index for index, code in enumerate(sorted_prdtypecode_list)}

st.title("Démo")

class CombinedModel(nn.Module):
    '''
    - desc : combine les entrées img et text et construit les couches de classification finale
    - params:
    -> text_input_size : taille du vecteur texte
    -> image_input_size : taille du vecteur image
    -> num_classes : nombre de classes en sortie de la couche de classification
    '''
    def __init__(self, text_input_size, image_input_size, num_classes):
        super(CombinedModel, self).__init__()
        # Deux couches fully connected (dense)
        self.fc1 = nn.Linear(text_input_size + image_input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.output = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, combined_features):
        # Concat des sorties des 2 model
        x = self.relu(self.fc1(combined_features))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.output(x)
        return x

class EcommerceDataset(Dataset):
    '''
    - desc : cette classe combine les model text, img, effectue une prediction pour chaque et retourne les features
    - params :
    -> df : données des produits
    -> text_model : modèle texte (chargé)
    -> img_model : modèle image (chargé)
    -> img_dir : path racine du dossier image
    - returns :
    -> text_features : sortie du modèle text
    -> img_features : sortie du modèle img
    -> label : label correspondant
    '''
    def __init__(self, df, text_model, img_model, img_dir):
        self.df = df
        self.text_model = text_model
        self.img_model = img_model
        self.img_dir = img_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Extraction text / img
        text_description = self.df.iloc[idx]['desi_desc_cleaned']
        prdtypecode = self.df.iloc[idx]['prdtypecode']
        image_name = self.df.iloc[idx]['image name']

        # path img
        image_path = os.path.join(self.img_dir, str(prdtypecode), image_name)
        
        # Chargement img
        img = Image.open(image_path)
        
        # Model prediction text - vecteur de probas
        text_features = self.text_model.predict_proba([text_description])[0]
        #print("type(text_features) : ", type(text_features))
        #print("len(text_features) : ", len(text_features))
        # Conversion en tensor si nécessaire
        if isinstance(text_features, np.ndarray):
            text_features = torch.tensor(text_features, dtype = torch.float32)
        #print("type(text_features) after tensor conv : ", type(text_features))
        #print("len(text_features) after tensor conv : ", len(text_features))
        
        # Prédictions du modèle image - vecteur de probas
        _, img_features = img_model.predict(image_path)
        #print("type(img_features) : ", type(img_features))
        #print("len(img_features) : ", len(img_features))
        # Conversion en tensor si nécessaire
        if isinstance(img_features, np.ndarray):
            img_features = torch.tensor(img_features, dtype = torch.float32)
        #print("type(img_features) after tensor conv : ", type(img_features))
        #print("len(img_features) after tensor conv : ", len(img_features))

        label = prdtypecode_dict[str(prdtypecode)]
        #print(f" In EcommerceDataset : Label: {label}, Type: {type(label)}, Size: {label.shape if hasattr(label, 'shape') else 'N/A'}")

        return text_features, img_features, label

model_combined_checkpoint = torch.load('/Users/ouissamgouni/Documents/it-workspace/bootcamp-mle-24/git/final/jul24_bds_rakuten/models/save/concatmodel_2024-10-07_08-52-25_epoch10of20.pth')
model_text = joblib.load('/Users/ouissamgouni/Documents/it-workspace/bootcamp-mle-24/git/final/jul24_bds_rakuten/models/save/finalized_model_text.sav')
model_img = torch.load('/Users/ouissamgouni/Documents/it-workspace/bootcamp-mle-24/git/final/jul24_bds_rakuten/models/save/finalized_model_img.pth')
img_model =model_img
model_c = CombinedModel(text_input_size = 27, image_input_size = 27, num_classes = NUM_CLASSES)
model_c.load_state_dict(model_combined_checkpoint)

model_c.eval()

_, val_df = train_test_split(df, test_size = 0.2, random_state = 42)

prdtypecode_sorted = ['10', '40', '50', '60', '1140', '1160', '1180', '1280', '1281', '1300', '1301', '1302', '1320', 
                    '1560', '1920', '1940', '2060', '2220', '2280', '2403', '2462', '2522', '2582', '2583', '2585', 
                    '2705', '2905']


samples_n = st.number_input("Insert a number of samples", value=10)
samples_per_col=3
rows_n = samples_n//samples_per_col + 1
rows = [st.columns(samples_per_col) for i in range(rows_n)]

cols = []
for r in rows:
    cols +=r

def fwrite_ok_ko(text, ok):
    if ok :
        ftext=":green-background["+text+"]"
    else:
        ftext=":red-background["+text+"]"
    st.markdown(ftext)

def fwrite_ok_ko_(text, ok):
        if ok :
            color='Green'
        else:
            color='Red'
        st.markdown('<p style="font-family:sans-serif; color:'+color+'; font-size: 10px;">'+ftext+'</p>', unsafe_allow_html=True)

with torch.no_grad():
    for i in range(samples_n):
        with cols[i]:
            with st.container(border=True):
                demo_df= val_df.sample(n=1)
                designation = demo_df.iloc[0]['designation']
                st.text('Designation: ' + str(designation))
                description = demo_df.iloc[0]['description']
                st.text('Description: ' + str(description))

                image_name = demo_df.iloc[0]['image name']
                prdtypecode = demo_df.iloc[0]['prdtypecode']
                image_path = os.path.join(DATA_DIR, str(prdtypecode), image_name)   
                img = Image.open(image_path)
                st.image(image=img)

                
                st.write('Expected product type:', prdtypecode,' ',demo_df.iloc[0]['désignation textuelle'])


                text_prediction = model_text.predict([demo_df.iloc[0]['desi_desc_cleaned']])[0]

                ftext="Text model :" + str(text_prediction) + ' ' + (df.loc[df['prdtypecode'] == int(text_prediction), 'désignation textuelle']).iloc[0]
                fwrite_ok_ko_(ftext, int(text_prediction) == prdtypecode)

                img_prediction, probabilities = img_model.predict(image_path)
                ftext="Image model :" + str(img_prediction) + ' ' + (df.loc[df['prdtypecode'] == int(img_prediction), 'désignation textuelle']).iloc[0]
                fwrite_ok_ko_(ftext, int(img_prediction) == prdtypecode)

                demo_dataset = EcommerceDataset(demo_df, model_text, model_img, DATA_DIR)
                demo_loader = DataLoader(demo_dataset, batch_size = 1, shuffle = False)
            
                for text_features, img_features, labels in demo_loader:
                    combined_features = torch.cat((text_features, img_features), dim = 1)
                    outputs = model_c(combined_features)
                    _, preds = torch.max(outputs, 1)
                    predicted_class = prdtypecode_sorted[preds.item()]
                    ftext="Final model :" + predicted_class + ' ' + (df.loc[df['prdtypecode'] == int(predicted_class), 'désignation textuelle']).iloc[0]
                    fwrite_ok_ko(ftext, int(predicted_class) == prdtypecode)
                    print('-'*20)
                