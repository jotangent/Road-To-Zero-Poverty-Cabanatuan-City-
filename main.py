import streamlit as st
import streamlit.components.v1 as components
import joblib

from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from matplotlib import pyplot as plt

st.set_page_config(layout="wide")
st.set_option('deprecation.showPyplotGlobalUse', False)

add_selectbox = st.sidebar.selectbox(
    "Topics",
    ("Introduction", "EDAs", "Poverty Classifier at Household Level", "Poverty Rate at Barangay Level", "Clustered Barangay")
)

loaded_model = joblib.load('lr_model.sav')
ml_reg = joblib.load('ml_reg.sav')
classification = joblib.load('classification.sav')
scaler = joblib.load('scaler.sav')

class_names = {"Poor","Not Poor"}

if add_selectbox == "Introduction":
    print("Introduction")    

elif add_selectbox == "EDAs":
    print("EDA")

elif add_selectbox == "Poverty Classifier at Household Level":
    header = st.beta_container()
    features = st.beta_container()


    ndeath05 = st.number_input(label="number of children who died ages 0-5", step=1)
    nmaln05 = st.number_input(label="number of children who experienced malnutrition ages 0-5", step=1)
    msh = st.text_input(label)

    # if st.button("Predict"):




        





        


    

        