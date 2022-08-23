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
import plotly.express as px

from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances, cosine_similarity



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
        
        
        
                
#                 [ndeath05, nntsws, ntert1721, nntliter10ab, nfshort, nvictcr, age_dep, dep]

#                 shapefile = shapefile.rename(columns = {'Infant Mor': 'Infant Mortality Deaths',
#                           'Malnourish': 'Malnourished Children 5 Yrs. Below', 
#                           'Living as': 'Living as Squatters',    
#                           'Living in': 'Living in Makeshift Housing',
#                           'No Access': 'No Access to Sanitary Toilet',
#                           'No Acces_1':'No Access to Safe Water',
#                           'Ages (5 an': 'Ages (5 and below) Not in Kinder',           
#                           'Ages (6-11': 'Ages (6-11) Not in Elementary',    
#                           'Ages (12-1': 'Ages (12-15) Not in Junior High School',
#                           #'nntelem612_prop': 'Ages (6-12) Not Attending Elementary', 
#                           #'nnths1316_prop': 'Ages (13-16) Not Attending Secondary',
#                           'Ages (16-1': 'Ages (16-17) Not Senior High School',                              
#                           'Ages (17-2': 'Ages (17-21) Not in Tertiary',
#                           'Ages (10 a': 'Ages (10 and above) Not Literate',
#                           'Poor House': 'Poor Household',    
#                           'Subsistent': 'Subsistently Poor Household',                
#                           'Experience': 'Experienced Food Shortage',                
#                           'Ages (15 a': 'Ages (15 and Above) Unemployed',
#                           'Number of':'Number of Victims of Crime',   
#                           'Dependents': 'Dependents  Ages (0-14, 65+)',
#                           'Unemployed': 'Unemployed Dependents',
#                           })            
           
        
# #         result_mlr_desc = '<p style="font-family:Arial; color:White; font-size: 20px;">Predicted Percentage of Poor Household in a Barangay is:</p>'   
# #         st.markdown(result_mlr_desc, unsafe_allow_html=True)
#         st.write("Predicted Percentage of Poor in Barangay is: ", str(("{:.2f}".format(float(result_mlr)))))
        
        
### Page 2 -------------------------------------------------------------      
        
        
elif my_page == 'Poverty Interactive Map':
    option1 = st.sidebar.selectbox(
    'View Selection', ['-- Please Select View --', 'One Barangay Only', 'All Barangays', 'Clusters'])
    
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
                
                
    if option1 == "Clusters":
        
        clusters_data = pd.read_csv('data/df_per_cluster_prop_rename.csv', index_col=[0])
        clusters_data = clusters_data.rename(columns = {'Infant Mortality Deaths': 'Infant Mortality Deaths', # or ndeath_prop?
                          'Malnourished Children 5 Yrs. Below': 'Malnourished Children 5 Yrs. Below', 
                          'Living as Squatters': 'Living as Squatters',    
                          'Living in Makeshift Housing': 'Living in Makeshift Housing',
                          'No Access to Sanitary Toilet Proportion': 'No Access to Sanitary Toilet',
                          'No Access to Safe Water Proportion':'No Access to Safe Water',
                          'Children Not in Kinder': 'Ages (5 and below) Not in Kinder',           
                          'Children Not in Elementary': 'Ages (6-11) Not in Elementary',    
                          'Children Not in Junior High School': 'Ages (12-15) Not in Junior High School',
                          #'nntelem612_prop': 'Ages (6-12) Not Attending Elementary', 
                          #'nnths1316_prop': 'Ages (13-16) Not Attending Secondary',
                          'Ages (16-17) Not Senior High School': 'Ages (16-17) Not Senior High School',                              
                          'Ages (17-21) Not in Tertiary': 'Ages (17-21) Not in Tertiary',
                          'Ages (10 and above) Not Literate': 'Ages (10 and above) Not Literate',
                          'Poor Household': 'Poor Household',     # ??
                          'nsubp_prop': 'Subsistently Poor Household',                
                          'Experienced food shortage': 'Experienced Food Shortage',                
                          'Unemployed 15 and above': 'Ages (15 and Above) Unemployed',
                          'Number of victims of crime':'Number of Victims of Crime',   
                          'Dependents (0-14, 65+)': 'Dependents  Ages (0-14, 65+)',
                          'Unemployed dependents': 'Unemployed Dependents',
                          })
        
        st.caption("2. Select Core Povery Indicator from Left Pane. There are 14 indicators available.")

        option1a = st.sidebar.selectbox(
        'Select Core Poverty Indicator',
            ['-- Please Select Poverty Core Indicator --'] + 
            prop_cols.drop(['barangay', 'cluster_labels'], axis = 1).columns.values.tolist())

        if option1a in prop_cols.columns.values.tolist():

            option_reg = st.sidebar.selectbox(
                'Select Cluster', ['-- Please select a cluster --', 'Security and Basic Education', 'Higher Education and Livelihood', 'Technical Opportunity & Child Care', 'Sanitation and Food Shortage'])

            if option_reg == '-- Please select a cluster --':
                st.caption("3. Select a cluster in from Left Pane. There are 4 clusters for selection.")

            else:
#                 heatmap_cluster_df = pd.DataFrame()
#                 heatmap_merged_data = pd.DataFrame()
                
                st.caption("3. Select a cluster from Left Pane. There are 4 clusters for selection.")
                st.caption("")
                st.markdown("<span style=' font-size: 25px'><span style='color:#ffbfbf'>" + option1a + "</span> in the <span style='color:#ffbfbf'>" + option_reg + '</span> cluster</span>', unsafe_allow_html=True)
                
                if option_reg == 'Security and Basic Education':
                    cluster_number = 3
                if option_reg == 'Higher Education and Livelihood':
                    cluster_number = 2
                if option_reg == 'Technical Opportunity & Child Care':
                    cluster_number = 1
                if option_reg == 'Sanitation and Food Shortage':
                    cluster_number = 0

                result = pd.DataFrame(clusters_data[clusters_data['cluster_labels'] == cluster_number][option1a])
                
                if option1a == 'Household Total Members':
                    st.caption("<span style=' font-size: 25px'>" + str(round(result.iloc[0,0], 4)) + "</span>", unsafe_allow_html=True)
                else:
                    st.caption("<span style=' font-size: 25px'>" + str(round(result.iloc[0,0], 4)) + "%</span>", unsafe_allow_html=True)
                    
                st.caption("")
                
                shapefile = shapefile.rename(columns = {'Household': 'Household Total Members'})
            
                #shortcut using shapefile only
                heatmap_filtered_data = shapefile[shapefile['cluster_la'] == cluster_number]
                heatmap_city_outline = shapefile[['barangay','geometry']]
                heatmap_city_outline['color'] = 0 
                heatmap_merged_data = pd.merge(heatmap_filtered_data, heatmap_city_outline, how='outer')
                heatmap_merged_data[option1a].fillna(0,inplace=True)
                
                range_min = heatmap_merged_data[option1a].min()
                if (heatmap_merged_data[option1a].max() == range_min):
                    range_max = range_min + 1
                else:
                    range_max = heatmap_merged_data[option1a].max()
                    
                heatmap_merged_data.set_index('barangay',inplace=True)
                fig = px.choropleth(heatmap_merged_data, geojson=heatmap_merged_data.geometry, 
                                    locations=heatmap_merged_data.index, color=option1a,height=500,color_continuous_scale="Oranges",
                                    range_color=[range_min, range_max])
    
                fig.update_geos(fitbounds="locations", visible=True)
                fig.update_layout(
#                     title_text=option_reg + ' Cluster:' + ' ' + option1a
                    title_text=''
                    
                )
                fig.update(layout = dict(title=dict(x=0.5)))
                fig.update_layout(
                    margin={"r":0,"t":30,"l":10,"b":10},
                    coloraxis_colorbar={
                        'title':'Percentage'})
                
                st.plotly_chart(fig)
                            
                
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
            folium_static(mymap)
            
            filtered_df = prop_cols.groupby('barangay').agg({option1b:'sum'})
            st.bar_chart(filtered_df, height = 400, width =2000)
            
            
   
        else:
            st.write("")
    
    
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