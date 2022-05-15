import numpy as np
import pandas as pd
import streamlit as st
from PIL import  Image
from datacleaning import DataCleaning


def app():
    display = Image.open('Logo.png')
    display = np.array(display)
    col1, col2 = st.columns(2)
    col1.image(display, width = 300)
    col2.title("Chronic Kidney Disease")
    #Page Title
    st.markdown('''
    # **Cleaned The Dataset:** ''')
    # Now Upload Dataset
    with st.header("Upload your dataset (.csv)"):
        upload_file = st.file_uploader("Upload the Dataset File:",type = ['csv'])

    if upload_file is not None:
        def load_csv():
            csv = pd.read_csv(upload_file)
            return csv
        dataset = load_csv()
        st.header("**Raw Dataset**")
        st.dataframe(dataset)
        if st.button("Submit"):

            clean = DataCleaning(dataset)
            clean.columns_rename()
            clean.replace_values()
            clean.categorcal_numerical()
            clean.wrong_datatypes()
            clean.nan_values()
            st.header("**Cleaned Dataset**")
            st.dataframe(dataset)
        # st.download_button(label="Download", data = dataset,file_name = "cleanData.csv")
        
            dataset.to_csv("Data\clean_data.csv",index = False)

    else:
        st.info('File Is Not Uploaded Yet!')

    




