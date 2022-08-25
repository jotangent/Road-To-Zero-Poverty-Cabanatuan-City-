import streamlit as st
import streamlit.components.v1 as components
import joblib
import lime
import shap
from lime.lime_tabular import LimeTabularExplainer
from sklearn.pipeline import make_pipeline
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
from PIL import Image
import plotly.express as px

import json
import folium
from streamlit_folium import folium_static
from folium.plugins import MarkerCluster

from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances, cosine_similarity

st.set_page_config(layout="wide")
st.set_option('deprecation.showPyplotGlobalUse', False)


prop_cols = pd.read_csv('data/df_prop_rename.csv', index_col=[0])
shapefile = gpd.read_file('data/Streamlit_Map/sl_map.shp')
shapefile["x"] = shapefile.geometry.centroid.x
shapefile["y"] = shapefile.geometry.centroid.y

shapefile = shapefile.rename(columns = {'Infant Mor': 'Infant Mortality Deaths',
                          'Malnourish': 'Malnourished Children 5 Yrs. Below', 
                          'Living as': 'Living as Squatters',    
                          'Living in': 'Living in Makeshift Housing',
                          'No Access': 'No Access to Sanitary Toilet',
                          'No Acces_1':'No Access to Safe Water',
                          'Ages (5 an': 'Ages (5 and below) Not in Kinder',           
                          'Ages (6-11': 'Ages (6-11) Not in Elementary',    
                          'Ages (12-1': 'Ages (12-15) Not in Junior High School',
                          #'nntelem612_prop': 'Ages (6-12) Not Attending Elementary', 
                          #'nnths1316_prop': 'Ages (13-16) Not Attending Secondary',
                          'Ages (16-1': 'Ages (16-17) Not Senior High School',                              
                          'Ages (17-2': 'Ages (17-21) Not in Tertiary',
                          'Ages (10 a': 'Ages (10 and above) Not Literate',
                          'Poor House': 'Poor Household',    
                          'Subsistent': 'Subsistently Poor Household',                
                          'Experience': 'Experienced Food Shortage',                
                          'Ages (15 a': 'Ages (15 and Above) Unemployed',
                          'Number of':'Number of Victims of Crime',   
                          'Dependents': 'Dependents  Ages (0-14, 65+)',
                          'Unemployed': 'Unemployed Dependents',
                          })


#-------- PAGE NAVIGATION

my_page = st.sidebar.radio('Page Navigation', 
['About the Project','Poverty Classifier at Household Level','Multiple Linear Regression', 'Poverty Interactive Map','Road Map to Alleviating Poverty']
)




#-------- FIRST PAGE --- ABOUT THE PROJECT --------------------------

if my_page == "About the Project":

    header = st.container()
    markdown = st.container()

    with header:
        st.header("Road To Zero Poverty: A machine learning approach to alleviating poverty in Cabanatuan City")

    with markdown:
        c1,c2,c3 = st.columns([1,5,1])
        st.markdown("""<div style="text-align: justify;">
        There are a lot of things that can influence a person's everyday life. From one's health and nutrition, to their community's peace and order, experiences could vastly differ. That being said, we, as data scientists, would like to understand this in a much deeper and quantifiable manner. This is where we would like to introduce the CBMS, or the Community Based Monitoring System, 
        which is an organized technology-based system of collecting, processing and validating data. This may be used for planning, program implementation and impact monitoring at the local level, 
        while empowering communities to participate in the process.
        </div>""", unsafe_allow_html=True)
        st.markdown('<p style="font-size:12px"><b>source: https://psa.gov.ph/content/community-based-monitoring-system-act<br></b></p>', unsafe_allow_html=True)

        st.markdown("""<div style="text-align: justify;">
        We were fortunate to have access to CBMS data from an LGU, particularly from Cabanatuan, and were asked to analyze and share insights. 
        We wanted our analysis to be in line with the UN's Sustainable Development Goals or SDGs, with emphasis to SDG #1 which is No Poverty.
        </div>""", unsafe_allow_html=True)
        
        st.markdown("""<div style="text-align: justify;"><br>
        The Community-Based Monitoring System (CBMS) is designed to cover various dimensions in characterizing the poverty situation at the local government level.
         The data gathered is intended to be utilized in developing a local program that includes social services and opportunities towards improving the quality of life for all. The CBMS census was conducted last 2018 in Cabanatuan and the results were unfortunately not retired by the PSA. 
        <br></div>""", unsafe_allow_html=True)

        st.subheader("""
        The Dataset
        """)

        st.markdown("""<div style="text-align: justify;">
        The dataset initially has 71,880 rows and 238 columns. After data cleaning, the rows have been decreased to 68,183 households and 38 features are deemed relevant for this study. 
        The classification whether a household is poor or not is based on the income threshold - a feature already identified in the dataset.
        </div>""", unsafe_allow_html=True)

        df = pd.read_csv('data/df_cleaned_removed_outliers.csv', encoding='latin-1')
        columns_df = pd.read_csv('data/column_details.csv')
        df = df.drop('Unnamed: 0', axis=1)


        st.dataframe(df)

        # --------------EDA-----------------
        st.subheader("""
        Exploratory Data Analysis
        """)

        st.markdown("""<div style="text-align: justify;">
        The group investigated the dimensions of poverty identified by CBMS through various EDA tools. Looking at the age demographics, 66% are in the labor force (residents ages 15-64). 
        Based on class composition, poverty is most prevalent in the young age group (ages 0-14) implying that poor households tend to have more children.
        </div>""", unsafe_allow_html=True)

        c1,c2,c3 = st.columns([2,5,2])
        with c1:
            st.write("")
        with c2:
            image = Image.open('pictures/ppt_EDA_age.png')
            st.image(image)
        with c3:
            st.write("")

        st.markdown("""<div style="text-align: justify;">
        Zooming in to other poverty indicators:
        </div>""", unsafe_allow_html=True)

        c1,c2,c3 = st.columns([2,5,2])
        with c1:
            st.write("")
        with c2:
            image = Image.open('pictures/ppt_EDA_poverty_indicators.png')
            st.image(image)
        with c3:
            st.write("")

        # HEALTH AND NUTRITION
        st.markdown("""<div style="text-align: justify;"><br>
        <b>1. Health and Nutrition:</b>
        <br>
        For this domain, CBMS is primarily investigating children under 5 years old who died, 
        women deaths due to pregnancy-related causes (only 3 cases have been reported in the entire city - 2 of which belongs to poor households), and children 0-5 years old who are malnourished.
        </div>""", unsafe_allow_html=True)

        c1,c2,c3 = st.columns([2,5,2])
        
        with c1:
            st.write("")
        with c2:
            image = Image.open('pictures/ppt_EDA_health_and_nutrition.png')
            st.image(image)
        with c3:
            st.write("")

        # HOUSING
        st.markdown("""<div style="text-align: justify;"><br>
        <b>2. Living Conditions </b>
        <br>
        The housing indicator includes households living in makeshift housing, and households classified as informal settler. 
        The census also accounts for water and sanitation or those households without access to safe water supply and households without access to sanitary toilet facilities.
        </div>""", unsafe_allow_html=True)

        c1,c2,c3 = st.columns([2,5,2])
        
        with c1:
            st.write("")
        with c2:
            image = Image.open('pictures/ppt_EDA_living_condition.png')
            st.image(image)
        with c3:
            st.write("")

        st.markdown("""<div style="text-align: justify;"><br>
        <b>3. Basic Education </b>
        <br>
        Basic education poverty covers the extent of children 6-11 years old who are not in elementary school, and children 12-15 years old who are not in secondary school.
        </div>""", unsafe_allow_html=True)

        c1,c2,c3 = st.columns([2,5,2])
        
        with c1:
            st.write("")
        with c2:
            image = Image.open('pictures/ppt_EDA_education.png')
            st.image(image)
        with c3:
            st.write("")
        
        st.markdown("""<div style="text-align: justify;"><br>
        <b>4. Income </b>
        <br>
        The income below the poverty threshold is typically the proxy variable in classifying poor households. In CBMS, this is also the variable used in tagging whether a household is poor or not. 
        Other indicators in this indicator are households with income below the food threshold and households that experienced hunger due to food shortage.
        </div>""", unsafe_allow_html=True)

        c1,c2,c3 = st.columns([2,5,2])
        
        with c1:
            st.write("")
        with c2:
            image = Image.open('pictures/ppt_EDA_prop_poor.png')
            st.image(image)
        with c3:
            st.write("")

        st.markdown("""<div style="text-align: justify;"><br>
        <b>5. Employment </b>
        <br>
        This portion tackles the part of the labor force that are employed. Note, however, that there is no specifics as to what is the nature of their employment or source of living.
        </div>""", unsafe_allow_html=True)

        c1,c2,c3 = st.columns([2,5,2])
        
        with c1:
            st.write("")
        with c2:
            image = Image.open('pictures/ppt_EDA_employment.png')
            st.image(image)
        with c3:
            st.write("")

        st.markdown("""<div style="text-align: justify;"><br>
        <b>6. Peace and Order </b>
        <br>
        The questionnaire for Peace and Order gathers whether the person has been a victim of crime. Its limitation is it does not specify the nature of the crime.
        </div>""", unsafe_allow_html=True)

        c1,c2,c3 = st.columns([1,8,1])
        
        with c1:
            st.write("")
        with c2:
            image = Image.open('pictures/ppt_EDA_peace_and_order2.png')
            st.image(image)
        with c3:
            st.write("")

###-----------OBJECTIVES---------------------
        st.subheader("Objectives")

        c1,c2,c3 = st.columns([2,5,2])
        
        with c1:
            st.write("")
        with c2:
            image = Image.open('pictures/ppt_Objectives.png')
            st.image(image)
        with c3:
            st.write("")

##-----------ML PIPELINE ------------------------------
        st.subheader("Machine Learning Pipeline")

        st.markdown("""<div style="text-align: justify;">
        Each model answers one objective. The Household Poverty Classifier helps uncover the factors affecting poverty at household level, the Multiple Linear Regressions quantify the factors with most impact in
        povert at barangay level, and the clustering helped us to strategically allocation the solution throughout Cabanatuan City. Below are the machine learning pipeline used in conducting these models.
        </div>""", unsafe_allow_html=True)

        st.markdown("""<div style="text-align: justify;"><p style="font-size:20px"><br>
        <b>Classification Model</b>        
        </div>""", unsafe_allow_html=True)                       

        c1,c2,c3 = st.columns([1,10,1])
        with c1:
            st.write("")

        with c2: 
            image = Image.open('pictures/classification_pipeline.png')
            st.image(image)
        
        with c3:
            st.write("")

        st.markdown("""<div style="text-align: justify;"><p style="font-size:20px"><br>
        <b>Clustering and Multiple Linear Regression</b>        
        </div>""", unsafe_allow_html=True)                       

        c1,c2,c3 = st.columns([1,10,1])
        with c1:
            st.write("")

        with c2: 
            image = Image.open('pictures/MLR_pipeline.png')
            st.image(image)  
        
        with c3:
            st.write("") 
              


       
#-------- SECOND PAGE --- POVERTY CLASSFIER --------------------------

elif my_page == "Poverty Classifier at Household Level":

    #------- INSERTING CLASSIFICATION MODEL - SCALER
    ml_reg = joblib.load('ml_reg.sav')
    classification = joblib.load('classification.sav')
    scaler = joblib.load('scaler.sav')
    class_names = {"Not Poor","Poor"}


    header = st.container()
    features = st.container()

    #---------- MODEL DESCRIPTION WITH SHAP IMAGE
    with header:
        st.header("Household Poverty Classifier")
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
        c1,c2,c3 = st.columns([1,3,1])
        with c1:
            st.write("")
        with c2:
            shap_png = Image.open('pictures/shap.png')
            st.image(shap_png)
        with c3:
            st.write("")
        st.markdown(" \n\n ")

    #---------- INSERT VALUES-----------
    with features:
        st.subheader("Please input the necessary features of a household")

        c1, c2 = st.columns(2)
        
        #NOT ATTENDING HIGHSCHOOL----------------------start
        nnths1215_description = '''<p style="font-family:Arial; font-size: 15px;"><b>
        Secondary Education
        </b></p>'''
        c1.markdown(nnths1215_description, unsafe_allow_html=True)
        nnths1215 = c1.number_input(label="number of teens not attending secondary school ages 12-15", step=1)
        #NOT ATTENDING HIGHSCHOOL----------------------end

        #NOT ATTENDING TERTIARY-------------------------start
        ntert1721_description = '''<p style="font-family:Arial; font-size: 15px;"><b>
        Tertiary Education
        </b></p>'''
        c1.markdown(ntert1721_description, unsafe_allow_html=True)
        ntert1721 = c1.number_input(label="number of adults not attending secondary school ages 17-21", step=1)
        #NOT ATTENDING TERTIARY-------------------------end

        #FOOD SHORTAGE-------------------------start
        fshort_description = '''<p style="font-family:Arial; font-size: 15px;"><b>
        Food Shortage
        </b></p>'''
        c1.markdown(fshort_description, unsafe_allow_html=True)
        fshort_option = c1.selectbox('have experienced food shortage?', options=['no', 'yes'])
        if fshort_option == 'yes':
            fshort = 1
        else:
            fshort = 0
        #FOOD SHORTAGE-------------------------end

        #UNEMPLOYMENT-------------------------start
        nunempl15ab_description = '''<p style="font-family:Arial; font-size: 15px;"><b>
        Unemployment
        </b></p>'''
        c1.markdown(nunempl15ab_description, unsafe_allow_html=True)
        nunempl15ab = c1.number_input(label="number of unemployed ages 15 and above", step=1)
        #UNEMPLOYMENT-------------------------end

        #Informal Settler--------------------------start
        squat_description = '''<p style="font-family:Arial; font-size: 15px;"><b>
        Informal Settler
        </b></p>'''
        c1.markdown(squat_description, unsafe_allow_html=True)
        squat_option = c1.selectbox('is the family informal settler?', options=['no', 'yes'])
        if squat_option == 'yes':
            squat = 1
        else:
            squat = 0
        #Informal Settler--------------------------end

        # ---------------------------- END OF C1

        

        #LABOR FORCE-------------------------start
        nlabfor_description = '''<p style="font-family:Arial; font-size: 15px;"><b>
        Labor Force
        </b></p>'''
        c2.markdown(nlabfor_description, unsafe_allow_html=True)
        nlabfor = c2.number_input(label="number of members that are eligible to work (ages 15 and above)", step=1)
        #LABOR FORCE-------------------------end

        #NMEM014-------------------------start
        nmem014_description = '''<p style="font-family:Arial; font-size: 15px;"><b>
        Age Dependents (children)
        </b></p>'''
        c2.markdown(nmem014_description, unsafe_allow_html=True)
        nmem014 = c2.number_input(label="number of members ages 0-14", step=1)
        #NMEM014-------------------------end    

        #NMEM65ab-------------------------start
        nmem65ab_description = '''<p style="font-family:Arial; font-size: 15px;"><b>
        Age Dependents (Senior)
        </b></p>'''
        c2.markdown(nmem65ab_description, unsafe_allow_html=True)
        nmem65ab = c2.number_input(label="number of members ages 65 and above", step=1)
        #NMEM65ab-------------------------end

        #NMEM1564-------------------------start
        nmem1564_description = '''<p style="font-family:Arial; font-size: 15px;"><b>
        Age Independent
        </b></p>'''
        c2.markdown(nmem1564_description, unsafe_allow_html=True)
        nmem1564 = c2.number_input(label="number of members ages 15-64", step=1)
        #NMEM1564-------------------------end

        #MAKESHIFT HOUSING--------------------------start
        msh_description = '''<p style="font-family:Arial; font-size: 15px;"><b>
        Makeshift Housing
        </b></p>'''
        c2.markdown(msh_description, unsafe_allow_html=True)
        msh_option = c2.selectbox('living in makeshift housing?', options=['no', 'yes'])
        if msh_option == 'yes':
            msh = 1
        else:
            msh = 0
        #MAKESHIFT HOUSING--------------------------end

        to_predict = [msh,squat,nnths1215,ntert1721,fshort,
                nunempl15ab,nlabfor,nmem014,nmem65ab,nmem1564]

        # ------------------------------- end of c2


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
            
        
        

        #---------LIME FOR EXPLAINABILITY
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






#-------- THIRD PAGE --- MULTIPLE LINEAR REGRESSION --------------------------
elif my_page == 'Multiple Linear Regression':

    ### Make Image Center
    st.header("Looking into Poverty at Barangay Level")
    # col1, col2, col3 = st.columns([1,6,1])
    # with col1:
    #     st.write("")
    # with col2:
    #     image = Image.open('pictures/MLR_Cabanatuan.PNG')
    #     st.image(image)
    # with col3:
    #     st.write("")

    st.markdown("""
    <b>Multiple Linear Regression (MLR)</b> was used to explore the linear approximate relationship of proportion
    of features to the proportion of poverty at barangay level in Cabanatuan City. The household data set as shown in the
    "About the Project" page was transformed to barangay dataset. The transformation involves getting the percentage of
    each column based on the columns's population, for example, the percentage of child death was computed by getting
    the sum of all the child death per barangay divided by the population of children in that particular barangay multiplied by 100.
    The barangay dataset is shown below for your reference.<br>""", unsafe_allow_html=True)

    df_per_barangay = pd.read_csv("data/df_per_barangay_final.csv")
    st.dataframe(df_per_barangay)

    st.markdown("""
    The trained model were evaluated through the following metrics: <br>
    """, unsafe_allow_html=True)
    c1,c2,c3 = st.columns([4,6,4])
    with c1:
        st.write("")
    with c2:
        image = Image.open('pictures/MLR_score.drawio.png')
        st.image(image)
    with c3:
        st.write("")
    
    st.markdown("""
    R-squared of the model is 0.61 which is greater than the 0.05 R-squared that is acceptable in social sciences. Root Mean Squared Error and Mean Average Error shows the average distances of the residuals from the
    fitted response.
    <br>
    The residual vs fitted response plot on the left shows that the model overestimates the poverty proportion when the fitted response is greater than 50%. While, the normal probability plot on the right
    shows that the residuals follow normal distribution which is acceptable in linear regression model.
    """, unsafe_allow_html=True)

    c1,c2 = st.columns([1,1])
    with c1:
        image = Image.open('pictures/MLR_residual_v_fitted.png')
        st.image(image)
    with c2:
        image = Image.open('pictures/MLR_prob_plot.png')
        st.image(image)

    st.markdown("""
    With these evaluation, the trained MLR model is only reliable when the predicted poverty proportion in a barangay is within 10 to 50%
    <br>
    <br>
    You can try the MLR model to explore the increase and decrease of poverty proportion with respect to the relevant features of a barangay below.
    """, unsafe_allow_html=True)


    st.title("Input Desired Percentage Value in Brgy. Level")
    st.caption("Values are automatically in % unit. No need to type %.")
    
# #testing only for df and shp file appearance in streamlit
#     st.title(prop_cols.columns) #this is a df
#     st.title(shapefile.columns) #this is a shp file

    
    ### Input values to X variables of MLR
    st.title("")
    
    col1, col2 = st.columns([1.2,1.2])
    with col1:
        ndeath05_desc = '<p style="font-family:Arial; font-size: 20px;"><b>Infant Mortality Ages 0-5</b></p>'
        st.markdown(ndeath05_desc, unsafe_allow_html=True) 
        st.caption("Proportion of Infant Mortality =  (No. of Infant Death Ages 0-5)  /  (Total Brgy. Population of Infant Ages 0-5)")
        ndeath05 = st.number_input('Max Value is 2.02%', min_value=0.0, max_value=2.02)
    with col2:
        nntsws_desc = '<p style="font-family:Arial; font-size: 20px;"><b>People with No Access to Safe Water</b></p>'
        st.markdown(nntsws_desc, unsafe_allow_html=True)
        st.caption("Proportion of No Access to Safe Water =  (No. of People with No Access to Safe Water)  /  (Total Brgy. Population)")
        nntsws = st.number_input('Max Value is 9.34%', min_value=0.0, max_value=9.34)
    
    
    st.subheader("")
    
    
    col1, col2 = st.columns([1.2,1.2])
    with col1:
        ntert1721_desc = '<p style="font-family:Arial; font-size: 20px;"><b>People that \
        are Not on Tertiary Ages 17-21</b></p>'
        st.markdown(ntert1721_desc, unsafe_allow_html=True)
        st.caption("Proportion of Ages 17-21 Not on Tertiary =  (No. of Ages 17-21 Not on Tertiary)  \
        /  (Total Brgy. Population of Ages 17-21)")
        ntert1721 = st.number_input('Max Value is 78.43%', min_value=0.0, max_value=78.43)
    with col2:
        nntliter10ab_desc = '<p style="font-family:Arial; font-size: 20px;"><b>People that are Not Literate</b></p>'
        st.markdown(nntliter10ab_desc, unsafe_allow_html=True)
        st.caption("Proportion of Illiterate Ages 10 above =  (No. of Illiterate Ages 10 above)  \
        /  (Total Brgy. Population of Ages 10 above)")
        nntliter10ab = st.number_input('Max Value is 12.58%', min_value=0.0, max_value=12.58)
    
    
    st.subheader("")
    
    
    col1, col2 = st.columns([1.2,1.2])
    with col1:
        nfshort_desc = '<p style="font-family:Arial; font-size: 20px;"><b>People that Experienced Food Shortage</b></p>'
        st.markdown(nfshort_desc, unsafe_allow_html=True)
        st.caption("Proportion of Experienced Food Shortage =  (No. of People Experienced Food Shortage)  /  (Total Brgy. Population)")
        nfshort = st.number_input('Max Value is 7.24', min_value=0.0, max_value=7.24)
    with col2:
        nvictcr_desc = '<p style="font-family:Arial; font-size: 20px;"><b>Victims of Crime</b></p>'
        st.markdown(nvictcr_desc, unsafe_allow_html=True)
        st.caption("Proportion of Victims of Crime =  (No. of Victims of Crime)  /  (Total Brgy. Population)")
        nvictcr = st.number_input('Max Value is 2.2', min_value=0.0, max_value=2.2)

    
    st.subheader("")
    
    
    col1, col2 = st.columns([1.2,1.2])
    with col1:
        age_dep_desc = '<p style="font-family:Arial; font-size: 20px;"><b>Dependents</b></p>'
        st.markdown(age_dep_desc, unsafe_allow_html=True)
        st.caption("Proportion of Dependents =  (No. of Dependent Ages 10-14) + (No. of Dependent Ages 65 above)  \
        /  (Total Brgy. Population)")
        st.caption("")
        age_dep = st.number_input('Max Value is 44.10', min_value=0.0, max_value=44.10)
    with col2:
        dep_desc = '<p style="font-family:Arial; font-size: 20px;"><b>Unemployed</b></p>'
        st.markdown(dep_desc, unsafe_allow_html=True)
        st.caption("Proportion of Unemployed =  [(Total Brgy. Population) - (Total No. of Labor Force) - \
        (Total No. of Unemployed 15 above)]  /  (Total Brgy. Population)")
        dep = st.number_input('Max Value is 84.84', min_value=0.0, max_value=84.84)
        
    
    st.subheader("")
    
    
    ###Store X variables in a list
    to_predict = [ndeath05, nntsws, ntert1721, nntliter10ab, nfshort, nvictcr, age_dep, dep]
    to_predict = np.asarray(to_predict).reshape(1,-1)
    
    ###Predict Percentage of Poor Household in a Barangay
    if st.button("Get Percentage of Poor Household in Barangay"):
        loaded_model = joblib.load('mlr_model.sav')
        result_mlr = loaded_model.predict(to_predict)
        
        
        if float(result_mlr) < 0:
            col1, col2, col3 = st.columns([4,1,5.5])
            with col1:
                st.caption("")
                st.caption("")
                st.subheader("Predicted Percentage of Poor in Barangay is: ")
            with col2:
                result_mlr_final = 0.00
                st.title('0.00%')
            with col3:
                st.write("")
       
        else:
            ###Make result in center
            col1, col2, col3 = st.columns([4,1,5.5])
            with col1:
                st.caption("")
                st.caption("")
                st.subheader("Predicted Percentage of Poor in Barangay is: ")
            with col2:
                result_mlr_final = float(result_mlr)
                st.title(str(("{:.2f}".format(result_mlr_final))) + '%')
            with col3:
                st.write("")

### Recommender engine for Barangay using Cosine Similarity
        #Create a df for to_predict list
        to_predict_dict = {
            'Infant Mortality Deaths' : [ndeath05],
            'No Access to Safe Water' : [nntsws],
            'Ages (17-21) Not in Tertiary' : [ntert1721],
            'Ages (10 and above) Not Literate' : [nntliter10ab],
            'Experienced Food Shortage' : [nfshort],
            'Number of Victims of Crime' : [nvictcr],
            'Dependents  Ages (0-14, 65+)' : [age_dep],
            'Unemployed Dependents' : [dep],
            'Poor Household' : [result_mlr_final]
        }

        to_predict_df = pd.DataFrame(to_predict_dict)
        
#         st.dataframe(to_predict_df)

        
        feature_cols = ['Poor Household', 'Infant Mortality Deaths', 'No Access to Safe Water', 'Ages (17-21) Not in Tertiary', 
                        'Ages (10 and above) Not Literate', 'Experienced Food Shortage', 'Number of Victims of Crime', 
                        'Dependents  Ages (0-14, 65+)', 'Unemployed Dependents']
        
        prop_cols['Euclidean Distance'] = prop_cols.apply(lambda x: euclidean_distances(x[feature_cols].values.reshape(-1, 1),\
                                                                  to_predict_df[feature_cols].values.reshape(-1, 1))\
                                                                  .flatten()[0], axis=1)
        
        recommendation_df = prop_cols.sort_values('Euclidean Distance')[:1]
        
        
        st.caption("Below are the most similar barangays compared to the Predicted Percentage of Poor in Barangay and the input values.")
        st.caption("The closer the Euclidean Distance to zero means it has high similarity.")
        st.caption("")
        st.dataframe(recommendation_df[['barangay', 'Euclidean Distance'] + feature_cols])




#-------- FOURTH PAGE --- INTERACTIVE MAP --------------------------

### old code from gelo

# elif my_page == 'Poverty Interactive Map':
#     option1 = st.sidebar.selectbox(
#     'View Selection', ['-- Please Select View --', 'One Barangay Only', 'All Barangays'])
    
#     st.title("Interactive Cabanatuan Barangay Map")
#     st.caption("")
#     st.caption("Instructions:")
#     st.caption("1. Select View from Left Pane. Choose if you want to view a particular brgy. or all barangays.")
#     map_center = [15.47, 121.035]

    
#     if option1 == "One Barangay Only":
#         st.caption("2. Select Core Povery Indicator from Left Pane. There are 14 indicators available.")
        
#         option1a = st.sidebar.selectbox(
#         'Select Core Poverty Indicator',
#             ['-- Please Select Poverty Core Indicator --'] + 
#             prop_cols.drop(['barangay', 'cluster_labels'], axis = 1).columns.values.tolist())
        

        
#         if option1a in prop_cols.columns.values.tolist():
            
            
#             # Styling the map
#             mymap = folium.Map(location=map_center, height=700, width=1000, tiles="OpenStreetMap", zoom_start=12)
#             marker_cluster = MarkerCluster().add_to(mymap)

             
            
#             option_reg = st.sidebar.selectbox(
#                 'Select Brgy. in Cabanatuan', ['-- Please Select a Brgy. --'] + list(shapefile["barangay"].sort_values(ascending = True).unique()))
            
#             if option_reg == '-- Please Select a Brgy. --':
#                 st.caption("3. Select a Barangay in Cabanatuan from Left Pane. There are 89 barangays for selection.")
            
#             else:
#                 st.caption("3. Select a Barangay in Cabanatuan from Left Pane. There are 89 barangays for selection.")
                
#                 st.subheader('You selected: ' + option1a + ' & ' + 'Brgy. ' + option_reg)

#                 reg = option_reg
#                 df_reg = shapefile[shapefile["barangay"]==reg]

#                 for i in np.arange(len(df_reg)):
#                     lat = df_reg["y"].values[i]
#                     lon = df_reg["x"].values[i]
#                     name = option1a + ": " + str(shapefile[option1a][i])
#                     mymap = folium.Map(location=[lat - 0.03, lon + 0.05], height=700, width=1000, tiles="OpenStreetMap", zoom_start=12)
#                     marker_cluster = MarkerCluster().add_to(mymap)
#                     folium.Marker([lat, lon], popup= name, tooltip = name).add_to(marker_cluster)
#                     folium.Popup(parse_html = True, show = True)
#                 folium_static(mymap)
            

            

#     elif option1 == "All Barangays":
#         # Styling the map
#         mymap = folium.Map(location=map_center, height=700, width=1000, tiles="OpenStreetMap", zoom_start=12)
#         marker_cluster = MarkerCluster().add_to(mymap)
            
#         option1b = st.sidebar.selectbox(
#         'Select Poverty Core Indicator',
#             ['-- Please Poverty Core Indicator --'] + 
#             shapefile.drop(['OBJECTID', 'NAME', 'Shape_Leng', 'Shape_Area', 'x', 'y', 'NAME_REV', 'barangay', 'geometry'],
#                            axis = 1).columns.values.tolist())
        
#         if option1b in shapefile.columns.values.tolist():
            
#             st.subheader('You selected: All Barangays & ' + option1b)
            
#             for i in np.arange(len(shapefile)):
#                     lat = shapefile["y"][i]
#                     lon = shapefile["x"][i]
#                     name = option1b + ": " + str(shapefile[option1b][i]) + '<br> Brgy. Name: ' + str(shapefile['barangay'][i])
#                     folium.Marker([lat, lon], popup = name, tooltip = name).add_to(marker_cluster)
#             folium_static(mymap)

elif my_page == 'Poverty Interactive Map':
    option1 = st.sidebar.selectbox(
    'View Selection', ['-- Please Select View --', 'One Barangay Only', 'All Barangays'])
    
    st.title("Interactive Cabanatuan Barangay Map")
    st.caption("")
    st.caption("Instructions:")
    st.caption("1. Select View from Left Pane. Choose if you want to view a particular brgy. or all barangays.")
    map_center = [15.47, 121.035]


    
   

    
    if option1 == "One Barangay Only":
        st.caption("2. Select Core Povery Indicator from Left Pane. There are 14 indicators available.")
        
        option1a = st.sidebar.selectbox(
        'Select Core Poverty Indicator',
            ['-- Please Select Poverty Core Indicator --'] + 
            prop_cols.drop(['barangay', 'cluster_labels'], axis = 1).columns.values.tolist())
        

        
        if option1a in prop_cols.columns.values.tolist():
            
            
            # Styling the map
            mymap = folium.Map(location=map_center, height=700, width=1000, tiles="OpenStreetMap", zoom_start=12)
            marker_cluster = MarkerCluster().add_to(mymap)

             
            
            option_reg = st.sidebar.selectbox(
                'Select Brgy. in Cabanatuan', ['-- Please Select a Brgy. --'] + list(shapefile["barangay"].sort_values(ascending = True).unique()))
            
            if option_reg == '-- Please Select a Brgy. --':
                st.caption("3. Select a Barangay in Cabanatuan from Left Pane. There are 89 barangays for selection.")
            
            else:
                st.caption("3. Select a Barangay in Cabanatuan from Left Pane. There are 89 barangays for selection.")
                
                st.subheader('You selected: ' + option1a + ' & ' + 'Brgy. ' + option_reg)

                reg = option_reg
                df_reg = shapefile[shapefile["barangay"]==reg]

                for i in np.arange(len(df_reg)):
                    lat = df_reg["y"].values[i]
                    lon = df_reg["x"].values[i]
##changes to be added in main.py
                    name = option1a + ": " + str("{:.2f}".format(shapefile[option1a][i])) + "%"
                    mymap = folium.Map(location=[lat - 0.03, lon + 0.05], height=700, width=1000, tiles="OpenStreetMap", zoom_start=12)
                    marker_cluster = MarkerCluster().add_to(mymap)
                    folium.Marker([lat, lon], popup= name, tooltip = name).add_to(marker_cluster)
                    folium.Popup(parse_html = True, show = True)
                folium_static(mymap)
            

            

    elif option1 == "All Barangays":
        st.caption("2. Select Core Povery Indicator from Left Pane. There are 14 indicators available.")
        
        # Styling the map
        mymap = folium.Map(location=map_center, height=700, width=1000, tiles="OpenStreetMap", zoom_start=12)
        marker_cluster = MarkerCluster().add_to(mymap)
            
        option1b = st.sidebar.selectbox(
        'Select Poverty Core Indicator',
            ['-- Please Select Poverty Core Indicator --'] + 
            shapefile.drop(['OBJECTID', 'NAME', 'Shape_Leng', 'Shape_Area', 'x', 'y', 'NAME_REV', 'barangay', 'geometry', 'cluster_la', 'Household'],
                           axis = 1).columns.values.tolist())
        
        if option1b in shapefile.columns.values.tolist():
            
            st.subheader('You selected: All Barangays & ' + option1b)
            
            if option1b == 'Malnourished Children 5 Yrs. Below':
                st.write("insert write up")
            elif option1b == 'Living as Squatters':
                st.write("")
            elif option1b == 'Living in Makeshift Housing':
                st.write("")
            elif option1b == 'No Access to Sanitary Toilet':
                st.write("")
            elif option1b == 'No Access to Sanitary Toilet':
                st.write("")
            elif option1b == 'Ages (5 and below) Not in Kinder':
                st.write("")
            elif option1b == 'Ages (6-11) Not in Elementary':
                st.write("")
            elif option1b == 'Ages (12-15) Not in Junior High School':
                st.write("")
            elif option1b == 'Ages (16-17) Not Senior High School':
                st.write("")
            elif option1b == 'Ages (17-21) Not in Tertiary':
                st.write("")
            elif option1b == 'Ages (10 and above) Not Literate':
                st.write("")
            elif option1b == 'Poor Household':
                st.write("")
            elif option1b == 'Subsistently Poor Household':
                st.write("")
            elif option1b == 'Experienced Food Shortage':
                st.write("")
            elif option1b == 'Ages (15 and Above) Unemployed':
                st.write("")
            elif option1b == 'Number of Victims of Crime':
                st.write("")
            elif option1b == 'Dependents  Ages (0-14, 65+)':
                st.write("")
            elif option1b == 'Unemployed Dependents':
                st.write("")
                
            for i in np.arange(len(shapefile)):
                lat = shapefile["y"][i]
                lon = shapefile["x"][i]
##changes to be added in main.py
                name = option1b + ": " + str("{:.2f}".format(shapefile[option1b][i])) + '%' + '<br> Brgy. Name: ' + \
                str(shapefile['barangay'][i]) 
                    
                folium.Marker([lat, lon], popup = name, tooltip = name).add_to(marker_cluster)
            
            st.caption("Interactive map of geo points")
            folium_static(mymap)
            
            brgy_data = pd.read_csv('data/df_prop_json_match.csv')
            json_geo_data = json.load(open('data/cabanatuan_brgy.json'))
            
            
            cab_choro_map = folium.Map(location = [15.494598024981352, 120.970035904559], tiles='cartodbpositron', zoom_start=12)
            
            cab_choro_map.choropleth(
                geo_data = json_geo_data,
                data = brgy_data,
                columns = ['barangay', option1b],
                key_on = 'feature.properties.NAME_3',
                fill_color='YlGnBu',
                fill_opacity = 0.3,
                line_weight=2
            )
            
            st.caption(f"Heatmap of all barangays measured by {option1b}")
            folium_static(cab_choro_map)
            
            st.caption(f"Bar graphs of all barangays measured by {option1b}")
            filtered_df = prop_cols.groupby('barangay').agg({option1b:'sum'})
            st.bar_chart(filtered_df, height = 400, width =2000)
            
            
   
        else:
            st.write("")

elif my_page == 'Road Map to Alleviating Poverty':
    st.image('pictures/road_map.png')



        





        


    

