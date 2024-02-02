import os
import streamlit as st 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
import unidecode
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from wordcloud import WordCloud
import spacy
cleaned_data  = None




def preprocess_multilang(text):
    nlp = spacy.load('fr_core_news_sm')
    text = text
    
    # Supprimer la ponctuation en conservant les caractères spéciaux et les accents
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    
    # Remplacer les séquences de blancs par un seul espace
    text = re.sub(r'\s+', ' ', text)
    
        # Supprimer les accents
    text = unidecode.unidecode(text)
    # Supprimer les chiffres si nécessaire
    # text = re.sub(r'\d+', '', text)
    
    return text.strip()


def generate_language_wordclouds(clean_data):
    # Regroupe les textes par langue et les concatène
    texts_by_language = {}
    for language, group in clean_data.groupby('Language'):
        texts_by_language[language] = ' '.join(group['text'])

    # Génère et affiche un nuage de mots pour chaque langue
    for language, text in texts_by_language.items():
        print(f"Nuage de mots pour la langue : {language}")
        wc = WordCloud(background_color="black", max_words=20)
        wc.generate(text)  # Utilisez directement `text` ici
        plt.figure(figsize=(10, 8))
        plt.imshow(wc.recolor(colormap='viridis', random_state=17), alpha=0.98)
        plt.axis('off')
        plt.show()

# Fonctions pour loader le dataset

def load_dataset(folder_path):
    if folder_path is not None:
        try:
            df = pd.read_csv(folder_path)
        except Exception as e:
                st.error("Erreur lors de la lecture du fichier. Assurez-vous qu'il s'agit d'un fichier CSV ou Excel.")
                return None
        return df
    return None

# Fonction qui fait appel à la fonction load_datset
def call_dataset():
    st.title("Chargement Automatique du Dataset")

    # Charger automatiquement le dataset
    df = load_dataset("LanguageDetection.csv")
    
    if df is not None:
        st.write("Aperçu des données :")

         # L'utilisateur peut choisir le nombre de lignes à afficher
        nrows = st.slider("Nombre de lignes à afficher :", 1, len(df), 5)  # Min: 1, Max: nombre total de lignes, Default: 5
            
        st.write(df.head(nrows))  # Affiche les n premières lignes du DataFrame en fonction du choix de l'utilisateur


# Fonction pour nettoyer, afficher le dataset nettoyé  et les nuages de mots
        
def analyse_data():
    # Chargement du modèle spacy pour le français
    nlp = spacy.load('fr_core_news_sm')    
    # Chargement du dataset
    df = load_dataset("LanguageDetection.csv")

    st.subheader("Nettoyage et affichage des 20 premières lignes du dataset")

    # Créez un bouton et vérifiez s'il a été cliqué
    if st.button("Affichez les 20 premières lignes du dataset nettoyé"):
        # Si le bouton est cliqué, effectuez le nettoyage
        cleaned_data = df.copy()
        # Application de la fonction de prétraitement sur la colonne 'Text'
        cleaned_data['text'] = df['Text'].apply(preprocess_multilang)
        # Affichage des 20 premières lignes du dataset nettoyé
        st.write(cleaned_data.head(20))
  
        # Affiche du nuage de mots
        texts_by_language = {}
        for language, group in clean_data.groupby('Language'):
            texts_by_language[language] = ' '.join(group['text'])

        # Génère et affiche un nuage de mots pour chaque langue
        for language, text in texts_by_language.items():
            st.write(f"Nuage de mots pour la langue : {language}")  # Utilisez st.write pour le texte
            wc = WordCloud(background_color="black", max_words=20)
            wc.generate(text)  # Utilisez directement `text` ici
            
            # Créer une figure pour le nuage de mots
            plt.figure(figsize=(10, 8))
            plt.imshow(wc.recolor(colormap='viridis', random_state=17), alpha=0.98)
            plt.axis('off')
            
            # Utilisez st.pyplot() pour afficher la figure dans Streamlit
            st.pyplot(plt)
            plt.close()       
