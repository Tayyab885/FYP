import numpy as np
import pandas as pd
import streamlit as st
from pages import utils
from PIL import  Image
import seaborn as sns
from pandas_profiling import ProfileReport, profile_report
from streamlit_pandas_profiling import st_profile_report

def app():
    display = Image.open('Logo.png')
    display = np.array(display)
    col1, col2 = st.columns(2)
    col1.image(display, width = 300)
    col2.title("Chronic Kidney Disease")
    #Page Title
    st.markdown('''
    # **Generate The Report Of The Dataset:** ''')

    # Now Upload Dataset
    with st.header("Upload your dataset (.csv)"):
        upload_file = st.file_uploader("Upload the Dataset File:",type = ['csv'])


    # Getting insights of Dataset
    if upload_file is not None:
        @st.cache
        def load_csv():
            csv = pd.read_csv(upload_file)
            return csv
        data = load_csv()
        report = ProfileReport(data,explorative = True)
        st.header("**Dataset**")
        st.write(data)
        st.write("---")
        st.header("**Dataset Report:**")
        st_profile_report(report)
    else:
        st.info('File is not Uploaded Yet!')



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


