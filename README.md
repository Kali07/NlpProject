# Modèle de Détection de Langue

## Description
Ce projet est un modèle de détection de la langue d'un texte basé sur un modèle d'apprentissage automatique. Il utilise un modèle régression logistique comme algorithme principal pré-entraîné pour prédire la langue du texte en entrée.

## Objectif
L'objectif principal de ce modèle est de détecter la langue d'un texte donné sur un lot de 17 langues : 
1) English
2) Malayalam
3) Hindi
4) Tamil
5) Kannada
6) French
7) Spanish
8) Portuguese
9) Italian
10) Russian
11) Sweedish
12) Dutch
13) Arabic
14) Turkish
15) German
16) Danish
17) Greek


## Algorithme
Nous avons utilisé le modèle de régression logistique pour entraîner ce modèle de détection de langue.

## Données
Les données d'entraînement et de test proviennent du corpus XYZ. Elles ont été prétraitées pour supprimer les caractères spéciaux et normaliser le texte.

## Instructions pour lancer la WebApp
1. Installez les dépendances en utilisant `pip install -r requirements.txt`.
2. Chargez le modèle et le vecteur TF-IDF à partir des fichiers .pkl en utilisant `joblib.load`.
3. Exécutez l'application Streamlit avec la commande `streamlit run app.py`.

## Exemple d'utilisation
Voici comment vous pouvez utiliser notre WebApp :
![Capture d'écran de l'application](screenshot.png)



## Auteur
Richard MULAMBA
Landry TOHOUNDO
Antoine VASONE
Ryan FONTAINE


