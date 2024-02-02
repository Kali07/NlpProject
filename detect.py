import streamlit as st
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from notebook_functions import preprocess_multilang




def load_model(model_path):
    model = joblib.load(model_path)
    return model

# Chemin vers le modèle
model_path = "rdfirstmodele.pkl"


model = load_model(model_path)


# Fonction pour charger du vectorizer
def load_vecto(vecto_path):
    # Utilisez joblib.load pour charger le modèle
    vecto = joblib.load(vecto_path)
    return vecto

vecto_path = "Tvecto.pkl"

# Utilisation de  la fonction pour charger le modèle
vecto = load_model(vecto_path)

def show_predict_page():
    st.title("Language Detection NLP App")
    st.subheader("Let's detect the Language of your sentence")
    
    menu = ["Home", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.subheader("Home")
        with st.form(key='nlpForm'):
            raw_text = st.text_area("Enter Text Here")
            submit_button = st.form_submit_button(label='Analyze')

        col1, col2 = st.columns(2)
        if submit_button:
            with col1:
                st.info("Cleaned Text")
                cleaned_text=preprocess_multilang(raw_text)
                st.write(cleaned_text)


            with col2:
                 st.info("Language")

                 vectorized_text = vecto.transform([cleaned_text])

                 detection = model.predict(vectorized_text)[0]  
                 st.write("Language:", detection)
