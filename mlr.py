from PIL import Image

from sklearn.decomposition import LatentDirichletAllocation
import streamlit as st
import streamlit.components.v1 as components
import joblib

import folium
from streamlit_folium import folium_static
from folium.plugins import MarkerCluster
from PIL import Image
import geopandas as gpd
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt




prop_cols = pd.read_csv('data/df_prop_rename.csv', index_col=[0])
shapefile = gpd.read_file('data/Streamlit_Map/sl_map.shp')
shapefile["x"] = shapefile.geometry.centroid.x
shapefile["y"] = shapefile.geometry.centroid.y


### ---------------------------------------------------------------
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

### ---------------------------------------------------------------


st.set_page_config(layout="wide")
st.set_option('deprecation.showPyplotGlobalUse', False)

# Actual Page
my_page = st.sidebar.radio('Page Navigation', ['Multiple Linear Regression', 'Poverty Interactive Map', 'About the Project'])



### Page 1 -------------------------------------------------------------


if my_page == 'Multiple Linear Regression':
    

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
            
           
        
# #         result_mlr_desc = '<p style="font-family:Arial; color:White; font-size: 20px;">Predicted Percentage of Poor Household in a Barangay is:</p>'   
# #         st.markdown(result_mlr_desc, unsafe_allow_html=True)
#         st.write("Predicted Percentage of Poor in Barangay is: ", str(("{:.2f}".format(float(result_mlr)))))
        
        
### Page 2 -------------------------------------------------------------      
        
        
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


    
    
# elif my_page == 'About the Project':

#         st.title("About the Project")
#         def page_slide():
#             components.html(
#         '<iframe src="https://docs.google.com/presentation/d/e/2PACX-1vT0BSqCK2IBpGqnPPJ2WkPTTMS3wSfjqTkBXRHz7ccYGrEyaatJ5EmCj9f_PPMACItZeyf1xUmyob5M/embed?start=false&loop=false&delayms=3000" frameborder="0" width="960" height="569" allowfullscreen="true" mozallowfullscreen="true" webkitallowfullscreen="true"></iframe>' 
#             ,height=1080, width=1920)

#         page_slide()

    
#         ### fill with slides ###
    
## Credits
# st.markdown("---")
# st.sidebar.markdown("
# The Team
# - Jac De Leon
# - Jota Dela Cruz
# - Jopet Fernandez
# - Ron Flores
# - Gelo Maandal

# Mentored by Patrisha Estrada                                                                                                               
# ")