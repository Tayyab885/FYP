import streamlit as st
import numpy as np
import pandas as pd
from pages import utils
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
        # df_visual = pd.DataFrame(df_analysis)
        # df_visual = df_analysis.copy()
        # cols = pd.read_csv('data/clean_data.csv')
        # Categorical,Numerical,Object = utils.getColumnTypes(cols)
        # cat_groups = {}
        # unique_Category_val={}

        # for i in range(len(Categorical)):
        #         unique_Category_val = {Categorical[i]: utils.mapunique(df_analysis, Categorical[i])}
        #         cat_groups = {Categorical[i]: df_visual.groupby(Categorical[i])}
        categorical = ['Red Blood Cells','Pus Cells','Pus Cell Clumps','Bacteria','Hypertension','Diabetes Mellitus','Coronary Artery Disease','Appetite','Pedal Edema','Anemia']
        
        category = st.selectbox("Select Category ", categorical )

        # sizes = (df_visual[category].value_counts()/df_visual[category].count())

        # labels = sizes.keys()

        # maxIndex = np.argmax(np.array(sizes))
        # explode = [0]*len(labels)
        # explode[int(maxIndex)] = 0.1
        # explode = tuple(explode)
        
        # fig1, ax1 = plt.subplots()
        # ax1.pie(sizes,explode = explode, labels=labels, autopct='%1.1f%%',shadow=False, startangle=0)
        # ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        # ax1.set_title('Distribution for Categorical Column - ' + (str)(category))
        # st.pyplot(fig1)
        
        # corr = df_analysis.corr(method='pearson')
        
        # fig2, ax2 = plt.subplots()
        # mask = np.zeros_like(corr, dtype=np.bool)
        # mask[np.triu_indices_from(mask)] = True
        # # Colors
        # cmap = sns.diverging_palette(240, 10, as_cmap=True)
        # sns.heatmap(corr, mask=mask, linewidths=.5, cmap=cmap, center=0,ax=ax2)
        # ax2.set_title("Correlation Matrix")
        # st.pyplot(fig2)
        
        
        # categoryObject=st.selectbox("Select " + (str)(category),unique_Category_val[category])
        # st.write(cat_groups[category].get_group(categoryObject).describe())
        # colName = st.selectbox("Select Column ",Numerical)

        # st.bar_chart(cat_groups[category].get_group(categoryObject)[colName])
        
        # ## Code base to drop redundent columns
        