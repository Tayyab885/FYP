import streamlit as st
import numpy as np
from PIL import Image
import webbrowser
from streamlit_option_menu import option_menu

def app():
#Row 1
    display = Image.open('Logo.png')
    display = np.array(display)
    col1, col2 = st.columns(2)
    col1.image(display,width = 300)
    col2.title("Chronic Kidney Disease")

    a1,a2 = st.columns(2)
    a1.title('Welcome!')
    a1.markdown('''Welcome to our Final Year Project(FYP).   
    Our Project is to Predict the Chronic Kidney          
    Disease(CKD) Using Data Science or Machine Learning.   
    In this project we will implement different Data Science   
    techinques to build a model to predict the Chronic Kidney   
    Disease.''')
    a2.image(Image.open('main.png'))

    #Row 2
    b1,b2 = st.columns(2)
    b1.markdown("""
    ****""")
    b1.markdown("""
    <style>
    .big-font {
        font-size:25px  !important;
        font-weight: bold; 
    }
    </style>
    """, unsafe_allow_html=True)

    b1.markdown('<p class="big-font">What is Chronic Kidney Disease(CKD)?</p>', unsafe_allow_html=True)
    url = "https://www.niddk.nih.gov/health-information/kidney-disease/chronic-kidney-disease-ckd/what-is-chronic-kidney-disease"
    if b1.button("Learn More"):
        webbrowser.open_new_tab(url)

    # hide_menu = '''
    # <style>
    # #MainMenu{
    #     visibility:hidden;
    #     }
    # footer{
    #     visibility:hidden;
    # }
    # </style>'''

    # st.markdown(hide_menu,unsafe_allow_html=True)

    b2.markdown('''\n\n
    Chronic kidney disease (CKD) means your kidneys are damaged and can’t filter blood the way they should. The disease is called “chronic” because the damage to your kidneys happens slowly over a long period of time. This damage can cause wastes to build up in your body. CKD can also cause other health problems.

    ​The kidneys’ main job is to filter extra water and wastes out of your blood to make urine. To keep your body working properly, the kidneys balance the salts and minerals—such as calcium, phosphorus, sodium, and potassium—that circulate in the blood. Your kidneys also make hormones that help control blood pressure, make red blood cells, and keep your bones strong.

    Kidney disease often can get worse over time and may lead to kidney failure. If your kidneys fail, you will need dialysis or a kidney transplant to maintain your health.

    The sooner you know you have kidney disease, the sooner you can make changes to protect your kidneys.''')
