import streamlit as st 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

# Models from Scikit-Learn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import requests
import os

st.set_page_config(
    page_title="Rainfall Prediction",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)

cwd = os.path.abspath('.')

def weatherApi_history_request():
    ###########-----History Area--------#######
    dates_list = ['2024-01-22', '2024-01-23', '2024-01-24', '2024-01-25', '2024-01-26', '2024-01-27', '2024-01-28', '2024-01-29']
    list_values = []
    list_columns = []
    for item in dates_list:
        url_history = "http://api.weatherapi.com/v1/history.json?key=1e9bd85382054a8493d123938240301&q=Athens&dt=" + item
        response_history = requests.get(url_history)
        response_json_history = response_history.json()
        for item in response_json_history['forecast']['forecastday'][0]['hour']:
            list_values.append(item)
            # create a dataframe with the values:
    df_history = pd.DataFrame(data = list_values)
    df_history = df_history[['time','temp_c','wind_kph','precip_mm','humidity','cloud', 'will_it_rain']]
    df_history.to_csv(cwd + "/weather_data/weatherApi_history.csv")
    
def weatherApi_current_request():
    url_forecast = "http://api.weatherapi.com/v1/forecast.json?key=1e9bd85382054a8493d123938240301&q=Athens&days=1&aqi=no&alerts=yes"
    
    ##########---Forecast Area-----##########
    response_forecast = requests.get(url_forecast)
    response_json_forecast = response_forecast.json()
    list_values = []
    list_columns = []
    for item in response_json_forecast['current'].items():
        list_columns.append(item)
        #list_values.append(response_json['current'][key])
    df_current = pd.DataFrame(data = list_columns)
    df_current = df_current.T
    df_current = df_current.rename(columns=df_current.iloc[0])
    df_current = df_current.drop(index=0)
    df_current = df_current.reset_index()
    df_current = df_current[['last_updated','temp_c','wind_kph','precip_mm','humidity','cloud']]
    #df_current = df_current.drop('index','last_updated_epoch','temp_f', 'condition', 'wind_mph', 'wind_degree', 'wind_dir', '', axis=1)
    df_current.to_csv(cwd + '/weather_data/weatherApi_current.csv')
    return df_current

# #############-------------MODELING-------------##############
@st.cache_data(persist="disk")
def modeling():
    df = pd.read_csv(cwd + '/weather_data/weatherApi_history.csv')#, skiprows=range(1, 3))
    
    df = df.drop(['Unnamed: 0'], axis=1)
    for row in range(0, len(df)):
        if df.loc[row, 'precip_mm'] == 0:
            df.loc[row, 'rain'] = 0
        elif df.loc[row, 'precip_mm'] != 0:
            df.loc[row, 'rain'] = 1

    X = df[['temp_c', 'wind_kph', 'humidity', 'cloud']] #,  'dew_point_2m', 'soil_temperature_0_to_7cm', 'cloud_cover_high']]
    y = df['rain']
    np.random.seed(42)
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.2)
    np.random.seed(42)
    #st.write("- Create the LogisticRegression Model")
    model = LogisticRegression()
    #st.write("- Fit the model")
    model.fit(X_train, y_train)    
    return model, df

def make_prediction(model):
    df_current = pd.read_csv(cwd + "/weather_data/weatherApi_current.csv")
    df_to_predict = df_current[['temp_c', 'wind_kph', 'humidity', 'cloud']]
    prediction = model.predict(df_to_predict)
    

    return df_to_predict, prediction, df_current
    
def main():
    st.sidebar.write("Hello")
    st.write("# Rainfall Weather Prediction")
    st.write("##### A LogisticRegression machine learning model!")
    
    model, df = modeling()
    st.write("#")
    if st.button('Request current weather data:'):
        df_current = weatherApi_current_request()
        st.success("Done!")

    if st.checkbox("Make current rainfal prediction:"):
        #Use the trained model to make predictions
        df_to_predict, prediction, df_current = make_prediction(model)
        col1, col2 = st.columns([3,1],gap='Large')
        with col1:
            st.info("Weather current report at date-time: ")
            st.dataframe(df_current, use_container_width=True)
        with col2:
            st.info("Current values to predict rain :")
            listdf = df_to_predict.values.tolist()
            # for item in listdf[0]:
            #     st.write(item)
            *listdf[0]

        st.write("### Prediction Results")
        if prediction == 0:
            st.write("**Prediction depending on current values: Will not be rain!**")
            if df_current['precip_mm'][0] != 0:
                st.write("**Current weather report rain is: Will be rain**")
                st.warning("The prediction on current values not matches with current weather report rain, is incorrect.")
            if df_current['precip_mm'][0] == 0:
                st.write("**Current weather report rain is: Will not be rain**")
                st.success("The prediction on current values matches with current weather report rain, is correct.")
        elif prediction==1:
            st.write("**Prediction depending on current values: Will be rain!**")
            if df_current['precip_mm'][0] != 0:
                st.write("**Current weather report rain is: Will be rain**")
                st.success("The prediction on current values matches with current weather report rain, is correct.")
            if df_current['precip_mm'][0] == 0:
                st.write("**Current weather report rain is: Will not be rain**")
                st.warning("The prediction on current values not matches with current weather report rain, is incorrect.")
    
    df_corr = df[['temp_c','wind_kph','precip_mm','humidity','cloud']]
    corr_matrix = df_corr.corr()
    corr_matrix_rain = df_corr.corr()['precip_mm']
    
if __name__ == '__main__':
    main()
