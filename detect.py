from langcodes import Language
import streamlit as st
import pickle
import joblib
import numpy as np
from notebook_functions import *


# Fonction pour charger du vectorizer
def load_vecto(vecto_path):
    vecto = joblib.load(vecto_path)
    return vecto

# Chemin vers le vectorizer
vecto_path = "Tvecto.pkl"

# Utilisation de  la fonction pour charger le vectorizer
vecto = load_vecto(vecto_path)




# Fonction pour charger le modèle
def load_model(model_path):
    # Utilisez joblib.load pour charger le modèle
    model = joblib.load(model_path)
    return model

# Chemin vers le modèle
model_path = "rdfirstmodele.pkl"

# Utilisation de  la fonction pour charger le modèle
model = load_model(model_path)



import streamlit as st

def show_predict_page():
    st.title("Language Detection NLP App")

    menu = ["Home", "Data Analytic"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.subheader("Home")
        st.subheader("Let's detect the Language of your sentence")

        with st.form(key='nlpForm'):
            raw_text = st.text_area("Enter Text Here")
            submit_button = st.form_submit_button(label='Analyze')

        if submit_button:
            col1, col2 = st.columns(2)

            with col1:
                st.info("Cleaned Text")
                cleaned_text = preprocess_multilang(raw_text)
                st.write(cleaned_text)

            with col2:
                st.info("Language")

                # Vectorisation - Notez que vous devriez utiliser 'transform' et non 'fit_transform' ici,
                # car 'fit_transform' est utilisé pour l'entraînement initial et ajuste le vectoriseur aux données fournies,
                # alors que 'transform' est utilisé pour transformer les données basées sur un ajustement existant.
                vectorized_text = vecto.transform([cleaned_text])

                # Detection de la langue
                detection = model.predict(vectorized_text)[0]
                st.write("Language:", detection)
    else:
        st.subheader("Let's analyse data")

      # Appel fonction pour loader et afficher les lignes du dataset charge par l'utilisateur 
        
        call_dataset()

      # Appel fonction pour nettoyer et afficher les 20 premieres lignes du dataset

        analyse_data()
