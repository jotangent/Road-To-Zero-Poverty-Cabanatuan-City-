import streamlit as st
import streamlit.components.v1 as components
import joblib
import lime
import shap
from lime.lime_tabular import LimeTabularExplainer
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

st.set_page_config(layout="wide")
st.set_option('deprecation.showPyplotGlobalUse', False)

add_selectbox = st.sidebar.selectbox(
    "NAVIGATE",
    ("Overview", "Methodology", "Poverty Classifier at Household Level", "Poverty Rate at Barangay Level", "Clustered Barangay")
)


ml_reg = joblib.load('ml_reg.sav')
classification = joblib.load('classification.sav')
scaler = joblib.load('scaler.sav')

class_names = {"Not Poor","Poor"}

if add_selectbox == "Introduction":
    st.header('Introduction')  

elif add_selectbox == "EDAs":
    st.header('EDAs')

elif add_selectbox == "Poverty Classifier at Household Level":
    header = st.container()
    features = st.container()

    with header:
        st.header("Welcome to our Household Poverty Classifier")
        st.markdown(
            
        """
        This is a GradientBoostingClassfier that classifies whether a household is poor or not based on specific features of a household in Cabanatuan City. 
        We trained the model using the CBMS 2018 census of Cabanatuan City. The household features below were selected by the multi-dimensions of poverty according to the CBMS 
        (i.e., health, nutrition, housing, water and sanitation, basic education, income, employment, and peace and order). 
        These dimensions were then narrowed down to ten based on our machine learning results. \n\n

        Considering all the households in Cabanatuan, the plot below shows how important each feature is to the decision making of the model. 
        To interpret the plot, the more positive the value of a point is, the more the model is leaning to classify a household as poor. 
        For example, if the number of dependents aged 0 to 14 is high, there is a higher probability that the household is poor. \n\n

        Based on the plot below, our model predicts poverty according to the ratio of working members to their dependents. 
        In short, the more mouths to feed on a small income, the more likely the household is to be classified as poor. 
        It can be seen that the largest feature to predicting a non-poor household is the number of working members. 
        The opposite is shown on other dimensions where scoring high tends to classify the household as poor.
        """
        )
        c1, c2, c3, c4 = st.columns(4)
        c1.image('shap.png', width = 1200)
        st.markdown(" \n\n ")

    with features:
        st.subheader("Please input the necessary features of a household")

        sel_col, disp_col = st.columns(2)

        msh_option = sel_col.selectbox('living in makeshift housing?', options=['no', 'yes'])
        if msh_option == 'yes':
            msh = 1
        else:
            msh = 0
        
        squat_option = sel_col.selectbox('is the family informal settler?', options=['no', 'yes'])
        if squat_option == 'yes':
            squat = 1
        else:
            squat = 0

        nnths1215 = sel_col.number_input(label="number of teens not attending secondary school ages 12-15", step=1)
        ntert1721 = sel_col.number_input(label="number of adults not attending secondary school ages 17-21", step=1)

        fshort_option = sel_col.selectbox('have experienced food shortage?', options=['no', 'yes'])
        if fshort_option == 'yes':
            fshort = 1
        else:
            fshort = 0

        nunempl15ab = disp_col.number_input(label="number of unemployed ages 15 and above", step=1)
        nlabfor = disp_col.number_input(label="number of members that are eligible to work (ages 15 and above)", step=1)
        nmem014 = disp_col.number_input(label="number of members ages 0-14", step=1)    
        nmem65ab = disp_col.number_input(label="number of members ages 65 and above", step=1)
        nmem1564 = disp_col.number_input(label="number of members ages 15-64", step=1)

        to_predict = [msh,squat,nnths1215,ntert1721,fshort,
                nunempl15ab,nlabfor,nmem014,nmem65ab,nmem1564]

        if st.button("PREDICT"):
            to_predict = np.asarray(to_predict).reshape(1,-1)
            to_predict = scaler.transform(to_predict)
            prediction = classification.predict(to_predict)
            if prediction == 1:
                st.header('The household is predicted as poor')
            else:
                st.header('The household is pretocted as not poor')
            
            st.markdown("""
            The model predicted whether the household with inputted properties is poor or not. To further explain the result, LIME Explainer was utilized.
            The prediction probabilities and the features integral to the model's prediction is listed below.
            """
            )
            
        
        

        #dataset
            df = pd.read_csv('data/df_cleaned_removed_outliers.csv', encoding='latin-1')
            columns_df = pd.read_csv('data/column_details.csv')

            df.drop(['Unnamed: 0','municipality'], axis=1, inplace=True)

            df = df.rename(columns = {'ndeath05': 'infant mortality deaths',
                          'nmaln05': 'malnourished children 5 below',
                          'msh': 'makeshift housing',
                          'squat': 'informal settlers',
                          'ntsws':'no access to safe water',
                          'ntstf': 'no access to sanitary toilet',
                          'nntelem611': 'number of children not in elementary',
                          'nnths1215': 'number of children not in secondary',
                          'ntert1721': 'number of members not in tertiary',
                          'fshort': 'experienced food shortage',
                          'nunempl15ab': 'number of unemployed 15 and above',
                          'nlabfor': 'number of members working',
                          'nmem014': 'dependents aged 0 to 14',
                          'nmem65ab': 'dependents aged 65 and above',
                          'nmem1564': 'independents aged 15 to 64',
                          'nvictcr':'number of victims of crime'})

            feature_cols = ['makeshift housing','informal settlers','number of children not in secondary','number of members not in tertiary',
                            'experienced food shortage','number of unemployed 15 and above', 'number of members working','dependents aged 0 to 14',
                            'dependents aged 65 and above', 'independents aged 15 to 64']

            X = scaler.transform(df[feature_cols])
            explainer = LimeTabularExplainer(X, mode="classification",
                                                class_names=[0, 1],
                                                feature_names=feature_cols,
                                                discretize_continuous=True
                                            )
            to_predict = to_predict[0]

            exp = explainer.explain_instance(to_predict, classification.predict_proba, num_features=10, top_labels=0)

            components.html(exp.as_html())                      






        





        


    

        