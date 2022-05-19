import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import  Image
import os

def app():
    display = Image.open('Logo.png')
    display = np.array(display)
    col1, col2 = st.columns(2)
    col1.image(display, width = 300)
    col2.title("Chronic Kidney Disease")

    if 'clean_data.csv' not in os.listdir('data'):
        st.markdown("Please upload data through `Upload Data` page!")
    else:
        df_analysis = pd.read_csv('data/clean_data.csv')
        categorical = ['Red Blood Cells','Pus Cells','Pus Cell Clumps','Bacteria','Hypertension','Diabetes Mellitus','Coronary Artery Disease','Appetite','Pedal Edema','Anemia']
        
        category = st.selectbox("Select Category ", categorical )



 












    hide_menu = '''
    <style>
    footer{
        visibility:hidden;
    }
    </style>'''

    st.markdown(hide_menu,unsafe_allow_html=True)