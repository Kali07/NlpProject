import os
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

def preprocess_multilang(text):
    text = text.lower()   
    
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    
    text = re.sub(r'\s+', ' ', text)
        # Supprimer les accents
    text = unidecode.unidecode(text)
    # Supprimer les chiffres si nécessaire
    # text = re.sub(r'\d+', '', text)
    
    return text.strip()