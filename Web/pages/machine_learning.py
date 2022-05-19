import os
import pandas as pd
import numpy as np
from PIL import Image
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

    st.markdown('''
    
    \n
    
    ''')
        # Load the data 
    if 'clean_data.csv' not in os.listdir('data'):
        st.markdown("Please upload data through `Upload Data` page!")
    else:
        data = pd.read_csv('data/clean_data.csv')
        y = data.iloc[:, -1]
        X = data[['Diabetes Mellitus', 'Specific Gravity', 'Hypertension',
       'Hemoglobin (gms)', 'Albumin', 'Packed Cell Volume', 'Appetite',
       'Red Blood Cells (millions/cmm)']]
    #Spliting Data into train and test

    X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.3, random_state=30)

    #Train Models
    m1,m2,m3 = st.columns(3)
    m1.header("Random Forest Tree")
    m1.text('''
    Random forest is a Supervised 
    Machine Learning Algorithm 
    that is used widely in 
    Classification and Regression 
    problems. It builds decision 
    trees on different samples and 
    takes their majority vote for 
    classification and average in 
    case of regression.''')
    if m1.button("RFT-Train/Test"):
        RandomForest = RandomForestClassifier()
        RandomForest = RandomForest.fit(X_train,y_train)
        y_pred = RandomForest.predict(X_test)
        st.write('Accuracy for Random Forest Tree:', accuracy_score(y_test,y_pred))

    m2.header("Gradient Boosting")
    m2.text('''
    Gradient boosting is a method 
    standing out for its prediction 
    speed and accuracy, particularly 
    with large and complex datasets. 
    From Kaggle competitions to machine 
    learning solutions for business, 
    this algorithm has produced 
    the best results.A gradient boosting 
    classifier is used when the target column is binary. 
    
    ''')
    if m2.button("GB-Train/Test"):
        GradientBoost = GradientBoostingClassifier()
        GradientBoost = GradientBoost.fit(X_train,y_train)
        y_pred = GradientBoost.predict(X_test)
        st.write('Accuracy for Gradient Boosting:', accuracy_score(y_test,y_pred))

    m3.header("Artificial NeuralNetwork(ANN)")
    m3.text('''
    Artificial Neural Network(ANN) 
    uses the processing of the brain 
    as a basis to develop algorithms 
    that can be used to model complex 
    patterns and prediction problems.
    ANNs are efficient data-driven 
    modelling tools widely used for 
    nonlinear systems dynamic modelling 
    and identification, due to their universal approximation capabilities and flexible structure that allow to capture complex nonlinear behaviors.
    ''')
    
    if m3.button("ANN-Train/Test"):
        model = Sequential()
        #first layer
        model.add(Dense(256,input_dim= len(X.columns),kernel_initializer= k.initializers.random_normal(seed= 13),activation= 'relu'))
        #second layer
        model.add(Dense (1, activation = 'hard_sigmoid'))
        model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
        #train the model
        ANN_model = model.fit(X_train, y_train, epochs = 200, batch_size = X_train.shape[0])
        st.write("Accuracy for ANN:",np.mean(ANN_model.history['accuracy']))


    hide_menu = '''
    <style>
    footer{
        visibility:hidden;
    }
    </style>'''

    st.markdown(hide_menu,unsafe_allow_html=True)