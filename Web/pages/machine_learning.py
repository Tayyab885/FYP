import os
import pandas as pd
import numpy as np
import streamlit as st

# Machine Learning 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import  DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
#ANN
import tensorflow as tf
import keras as k
from tensorflow import keras
from PIL import Image
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


def app():
    display = Image.open('Logo.png')
    display = np.array(display)
    col1, col2 = st.columns(2)
    col1.image(display,width = 300)
    col2.title("Chronic Kidney Disease")
    # Load the data 
    if 'clean_data.csv' not in os.listdir('data'):
        st.markdown("Please upload data through `Upload Data` page!")
    else:
        data = pd.read_csv('data/clean_data.csv')
        st.write(data)
       