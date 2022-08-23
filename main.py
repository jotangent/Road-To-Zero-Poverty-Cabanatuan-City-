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

shapefile = shapefile.rename(columns = {'ndeath05_prop': 'Infant mortality deaths prop',
                          'Malnourish': 'Malnourished Children 5 Yrs. Below Proportion',
                          'ntsws_prop':'No access to safe water',
                          'ntstf_prop': 'No access to sanitary toilet',
                          'nntelem611_prop': 'Children not in elementary',
                          'nnths1215_prop': 'Children not in secondary',
                          'ntert1721_prop': 'Children not in tertiary',
                          'fshort_prop': 'Experienced food shortage',
                          'nunempl15ab_prop': 'Unemployed 15 and above',
                          'nlabfor_prop': 'People working',
                          'nmem014_prop': 'Dependents aged 0 to 14',
                          'nmem65ab_prop': 'Dependents aged 65 and above',
                          'nmem1564_prop': 'Eligible working people',
                          'nvictcr_prop':'Number of victims of crime',
                          'nntsws_prop': 'No access to safe water',
                          'nntelem612_prop': 'Ages (6-12) not attending elementary',
                          'nnths1617_prop': 'Ages (16-17) not attending highschool',
                          'nntliter10ab_prop': 'Ages (10 and above) not literate',
                          'nfshort_prop': 'Experienced food shortage',
                          'age_dep_prop': 'Dependents (0-14, 65+)',
                          'dep_prop': 'Unemployed dependents',
                          'npovp_prop': 'number of poor'})


#-------- PAGE NAVIGATION

my_page = st.sidebar.radio('Page Navigation', 
['About the Project','Poverty Classifier at Household Level','Multiple Linear Regression', 'Poverty Interactive Map']
)




#-------- FIRST PAGE --- ABOUT THE PROJECT --------------------------

if my_page == "About the Project":





    header = st.container()
    markdown = st.container()

    with header:
        st.header("Road To Zero Poverty: A machine learning approach to alleviating poverty in Cabanatuan City")

    with markdown:
        col1, col2 = st.columns(2)
        st.markdown("""<div style="text-align: justify;">
        There are a lot of things that can influence a person's everyday life. From one's health and nutrition, to their community's peace and order, experiences could vastly differ. That being said, we, as data scientists, would like to understand this in a much deeper and quantifiable manner. This is where we would like to introduce the CBMS, or the Community Based Monitoring System, 
        which is an organized technology-based system of collecting, processing and validating data. This may be used for planning, program implementation and impact monitoring at the local level, 
        while empowering communities to participate in the process.
        </div>""", unsafe_allow_html=True)
        st.markdown('<p style="font-size:12px">source: https://psa.gov.ph/content/community-based-monitoring-system-act<br></p>', unsafe_allow_html=True)

        st.markdown("""<div style="text-align: justify;">
        We were fortunate to have access to CBMS data from an LGU, particularly from Cabanatuan, and were asked to analyze and share insights. 
        We wanted our analysis to be in line with the UN's Sustainable Development Goals or SDGs, with emphasis to SDG #1 which is No Poverty.
        """, unsafe_allow_html=True)

        st.markdown("""<div style="text-align: justify;"><br>
        But to have a future with no poverty, it is important to know the nature and extent of poverty: who the poor are, where they are, and why they are poor. 
        """, unsafe_allow_html=True)
        

















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

    prop_cols = pd.read_csv('data/df_prop_rename.csv', index_col=[0])
    shapefile = gpd.read_file('data/Streamlit_Map/sl_map.shp')
    shapefile["x"] = shapefile.geometry.centroid.x
    shapefile["y"] = shapefile.geometry.centroid.y


### Renaming of shapfile column names to match the prop_cols columns---------------------------------------------------------------
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

    ### Make Image Center
    col1, col2, col3 = st.columns([1,6,1])
    with col1:
        st.write("")
    with col2:
        image = Image.open('pictures/MLR_Cabanatuan.PNG')
        st.image(image)
    with col3:
        st.write("")
    
    st.subheader("")
    
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
        
        recommendation_df = prop_cols.sort_values('Euclidean Distance')[:3]
        
        
        st.caption("Below are the most similar barangays compared to the Predicted Percentage of Poor in Barangay and the input values.")
        st.caption("The closer the cosine distance to zero means it has high similarity.")
        st.caption("")
        st.dataframe(recommendation_df[['barangay', 'Euclidean Distance'] + feature_cols])



#-------- FOURTH PAGE --- INTERACTIVE MAP --------------------------

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
                    name = option1a + ": " + str(shapefile[option1a][i])
                    mymap = folium.Map(location=[lat - 0.03, lon + 0.05], height=700, width=1000, tiles="OpenStreetMap", zoom_start=12)
                    marker_cluster = MarkerCluster().add_to(mymap)
                    folium.Marker([lat, lon], popup= name, tooltip = name).add_to(marker_cluster)
                    folium.Popup(parse_html = True, show = True)
                folium_static(mymap)
            

            

    elif option1 == "All Barangays":
        # Styling the map
        mymap = folium.Map(location=map_center, height=700, width=1000, tiles="OpenStreetMap", zoom_start=12)
        marker_cluster = MarkerCluster().add_to(mymap)
            
        option1b = st.sidebar.selectbox(
        'Select Poverty Core Indicator',
            ['-- Please Poverty Core Indicator --'] + 
            shapefile.drop(['OBJECTID', 'NAME', 'Shape_Leng', 'Shape_Area', 'x', 'y', 'NAME_REV', 'barangay', 'geometry'],
                           axis = 1).columns.values.tolist())
        
        if option1b in shapefile.columns.values.tolist():
            
            st.subheader('You selected: All Barangays & ' + option1b)
            
            for i in np.arange(len(shapefile)):
                    lat = shapefile["y"][i]
                    lon = shapefile["x"][i]
                    name = option1b + ": " + str(shapefile[option1b][i]) + '<br> Brgy. Name: ' + str(shapefile['barangay'][i])
                    folium.Marker([lat, lon], popup = name, tooltip = name).add_to(marker_cluster)
            folium_static(mymap)




        





        


    

        