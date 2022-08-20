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



my_page = st.sidebar.radio('Page Navigation', 
['About the Project','Poverty Classifier at Household Level','Multiple Linear Regression', 'Poverty Interactive Map']
)


ml_reg = joblib.load('ml_reg.sav')
classification = joblib.load('classification.sav')
scaler = joblib.load('scaler.sav')

class_names = {"Not Poor","Poor"}


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
        



















elif my_page == "Poverty Classifier at Household Level":
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
        c1,c2,c3 = st.columns([1,3,1])
        with c1:
            st.write("")
        with c2:
            shap_png = Image.open('pictures/shap.png')
            st.image(shap_png)
        with c3:
            st.write("")
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


elif my_page == 'Multiple Linear Regression':

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
    
    st.title("Input Desired Percentage Value")
    st.caption("Values are automatically in % unit. No need to type %.")
    
    
    st.title(prop_cols.columns) #this is a df
    st.title(shapefile.columns) #this is a shp file

    
    ### Input values to X variables of MLR
    st.title("")
    
    col1, col2 = st.columns([1.2,1.2])
    with col1:
        ndeath05_desc = '<p style="font-family:Arial; color:White; font-size: 20px;"><b>Death of Children Ages 0-5</b></p>'
        st.markdown(ndeath05_desc, unsafe_allow_html=True) 
        ndeath05 = st.number_input('Max Value is 2.02%', min_value=0.0, max_value=2.02)
    with col2:
        nntsws_desc = '<p style="font-family:Arial; color:White; font-size: 20px;"><b>People with No Access to Safe Water</b></p>'
        st.markdown(nntsws_desc, unsafe_allow_html=True)
        nntsws = st.number_input('Max Value is 9.34%', min_value=0.0, max_value=9.34)
    
    
    st.subheader("")
    
    
    col1, col2 = st.columns([1.2,1.2])
    with col1:
        ntert1721_desc = '<p style="font-family:Arial; color:White; font-size: 20px;"><b>People that are Not on Tertiary Ages 17-21</b></p>'
        st.markdown(ntert1721_desc, unsafe_allow_html=True)
        ntert1721 = st.number_input('Max Value is 78.43%', min_value=0.0, max_value=78.43)
    with col2:
        nntliter10ab_desc = '<p style="font-family:Arial; color:White; font-size: 20px;"><b>People that are Not Literate</b></p>'
        st.markdown(nntliter10ab_desc, unsafe_allow_html=True)
        st.subheader("The proportion of Tertiary is calculated by...")
        nntliter10ab = st.number_input('Max Value is 12.58%', min_value=0.0, max_value=12.58)
    
    
    st.subheader("")
    
    
    col1, col2 = st.columns([1.2,1.2])
    with col1:
        nfshort_desc = '<p style="font-family:Arial; color:White; font-size: 20px;"><b>People that Experienced Food Shortage</b></p>'
        st.markdown(nfshort_desc, unsafe_allow_html=True)
        nfshort = st.number_input('Max Value is 7.24', min_value=0.0, max_value=7.24)
    with col2:
        nvictcr_desc = '<p style="font-family:Arial; color:White; font-size: 20px;"><b>Percentage of Victims of Crime</b></p>'
        st.markdown(nvictcr_desc, unsafe_allow_html=True)
        nvictcr = st.number_input('Max Value is 2.2', min_value=0.0, max_value=2.2)

    
    st.subheader("")
    
    
    col1, col2 = st.columns([1.2,1.2])
    with col1:
        age_dep_desc = '<p style="font-family:Arial; color:White; font-size: 20px;"><b>Percentage of Age Dependents</b></p>'
        st.markdown(age_dep_desc, unsafe_allow_html=True)
        age_dep = st.number_input('Max Value is 44.10', min_value=0.0, max_value=44.10)
    with col2:
        dep_desc = '<p style="font-family:Arial; color:White; font-size: 20px;"><b>Percentage of Unemployed</b></p>'
        st.markdown(dep_desc, unsafe_allow_html=True)
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
                st.title(str(("{:.2f}".format(float(result_mlr)))) + '%')
            with col3:
                st.write("")               


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




        





        


    

        