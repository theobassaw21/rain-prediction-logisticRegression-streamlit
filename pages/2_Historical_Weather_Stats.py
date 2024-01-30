import streamlit as st 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pickle
import seaborn as sb


st.set_page_config(
    page_title="Statistics",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)

@st.cache_data(persist="disk")
def read_csv():
    df = pd.read_csv('historical_data/hourly_till_now.csv', skiprows=range(1, 3))
    #df = df.drop(['Unnamed: 0'], axis=1)
    df["date"] = pd.to_datetime(df["date"])
    df['hour'] = df['date'].dt.hour

    df['day'] = df['date'].dt.strftime('%d')
    df['month'] = df['date'].dt.strftime('%m')
    df['year'] = df['date'].dt.strftime('%Y')
    df = df.drop('Unnamed: 0', axis=1)
    # convert to 0 or 1 the column rain depending of its values:
    for row in range(0, len(df)):
        if df.loc[row, 'rain'] == 0:
            df.loc[row, 'rain'] = 0
        elif df.loc[row, 'rain'] != 0:
            df.loc[row, 'rain'] = 1

    # create new column depending on five categories of temperature:
    for row in range(0, len(df)):
        if 0 > df.loc[row, 'temperature_2m']:
            df.loc[row, 'temperature_2m_category'] = 0
        elif 0 <= df.loc[row, 'temperature_2m'] < 10:
            df.loc[row, 'temperature_2m_category'] = 1
        elif 10 <= df.loc[row, 'temperature_2m'] < 20:
            df.loc[row, 'temperature_2m_category'] = 2
        elif 20 <= df.loc[row, 'temperature_2m'] < 30:
            df.loc[row, 'temperature_2m_category'] = 3
        elif 30 <= df.loc[row, 'temperature_2m'] < 40:
            df.loc[row, 'temperature_2m_category'] = 4
        else:
            df.loc[row, 'temperature_2m_category'] = 5
    len_df = len(df)
    return df, len_df


st.write("## Statistis on Weather Historical Data:")
st.write("#### Read the csv file with data:")
df, len_df = read_csv()

start_date = pd.Timestamp(st.sidebar.date_input("Start date", df['date'].min().date()))
end_date = pd.Timestamp(st.sidebar.date_input("End date", df['date'].max().date()))

st.write(df, use_container_width=True)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.write('- Total days:', len(df))
    
with col2:
    st.write("- Total days with rain", len(df['rain'][df['rain']==1]))
with col3:
    st.write("- Total days without rain", len(df['rain'][df['rain']==0]))
with col4:
    st.write("- Total days without snowfall", len(df['snowfall'][df['snowfall']>0]))
st.write("#")
st.write("""Created five categories for temperature to summarize the values. \n
    category 0 : where temperature is under 0 \n 
    category 1 : where temperature is between 0 and 10 \n
    category 2 : where temperature is between 10 and 20 \n
    category 3 : where temperature is between 20 and 30 \n
    category 4 : where temperature is between 30 and 40\n
    category 5 : where temperature is over 40 \n""")

st.write("#")

col0, col1, col2, col3, col4 ,col5= st.columns(6)
with col0:
    st.write("##### Temps under 0:")
    st.write("Total days:", len(df['temperature_2m_category'][df['temperature_2m_category']==0]))
    st.write("Total days with rain:", len(df['temperature_2m_category'][df['temperature_2m_category']==0][df['rain']==1]))
    st.write("Total days without rain:", len(df['temperature_2m_category'][df['temperature_2m_category']==0][df['rain']==0]))

with col1:
    st.write("##### Temps between 0 and 10:")
    st.write("Total days:", len(df['temperature_2m_category'][df['temperature_2m_category']==1]))
    st.write("Total days with rain:", len(df['temperature_2m_category'][df['temperature_2m_category']==1][df['rain']==1]))
    st.write("Total days without rain:", len(df['temperature_2m_category'][df['temperature_2m_category']==1][df['rain']==0]))

with col2:
    st.write("##### Temps between 10 and 20:")
    st.write("Total days:", len(df['temperature_2m_category'][df['temperature_2m_category']==2]))
    st.write("Total days with rain:", len(df['temperature_2m_category'][df['temperature_2m_category']==2][df['rain']==1]))
    st.write("Total days without rain:", len(df['temperature_2m_category'][df['temperature_2m_category']==2][df['rain']==0]))

with col3:
    st.write("##### Temps between 20 and 30:")
    st.write("Total days:", len(df['temperature_2m_category'][df['temperature_2m_category']==3]))
    st.write("Total days with rain:", len(df['temperature_2m_category'][df['temperature_2m_category']==3][df['rain']==1]))
    st.write("Total days without rain:", len(df['temperature_2m_category'][df['temperature_2m_category']==3][df['rain']==0]))

with col4:
    st.write("##### Temps between 30 and 40:")
    st.write("Total days:", len(df['temperature_2m_category'][df['temperature_2m_category']==4]))
    st.write("Total days with rain:", len(df['temperature_2m_category'][df['temperature_2m_category']==4][df['rain']==1]))
    st.write("Total days without rain:", len(df['temperature_2m_category'][df['temperature_2m_category']==4][df['rain']==0]))

with col5:
    st.write("##### Temps over 40:")
    st.write("Total days:", len(df['temperature_2m_category'][df['temperature_2m_category']==5]))
    st.write("Total days with rain:", len(df['temperature_2m_category'][df['temperature_2m_category']==5][df['rain']==1]))
    st.write("Total days without rain:", len(df['temperature_2m_category'][df['temperature_2m_category']==5][df['rain']==0]))


st.write("#")
col1 , col2 = st.columns(2, gap='Large')
with col1:
    st.write("##### Counts for each category of temperature:\n - Period:", start_date, " to", end_date)
    fig = px.pie(df, names='temperature_2m_category')
    fig.update_traces(hoverinfo='label+percent', textinfo='label+percent+value')
    fig.update_layout(
    
    )
    st.plotly_chart(fig, use_container_width=True)  

with col2:
    raining_days_count_per_month = (df['month'][df['rain']==1].value_counts() / 240)
    st.write("##### Average number of raining days per month\n - Period:", start_date, " to", end_date)

    
    fig = px.bar(df, x=raining_days_count_per_month.index , y= raining_days_count_per_month.values , width=900, height=500)
    fig.update_layout(margin=dict(l=20, r=20, t=50, b=20))
    fig.update_xaxes(rangemode='tozero', showgrid=False)
    fig.update_yaxes(rangemode='tozero', showgrid=True)
    st.plotly_chart(fig, use_container_width=True)

st.write("#")
st.write("##### Display data based on time duration - Period:", start_date, " to", end_date)
st.write("#### Stats when rain:")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.write("-Top 5 Months with most raining days:")
    top_months = raining_days_count_per_month.nlargest(n=5)
    st.write(top_months)
with col2:
    st.write("-Top hours list, that usally rain:")
    top_raining_hours = df['hour'][df['rain']==1].value_counts().nlargest(n=5,)
    st.write(top_raining_hours)
with col3:
    st.write("-Top hours list, that usally dont rain:")
    top_raining_hours = df['hour'][df['rain']==0].value_counts().nlargest(n=5,)
    st.write(top_raining_hours)
with col4:
    st.write("-Most hourly precipitation:")
    most_hourly_precipitation = df[['date', 'precipitation']].nlargest(n=5, columns=['precipitation'])
    st.write(most_hourly_precipitation)


st.write("---")
st.write("##### Display the sum of precipitation per year:\n - Period:", start_date, " to", end_date)
st.write("#")

df_tmp = df['precipitation'].groupby(df.year).sum()
fig = px.bar(df_tmp, y="precipitation" , width=900, height=500,
labels={
                     "df['precipitation'].groupby(df.year).sum().index": "Sepal Length (cm)",
                     "df['precipitation'].groupby(df.year).sum().values": "Sepal Width (cm)",
                     
                 })
fig.update_layout(
    title=dict(text="Precipitation-per-year", font=dict(size=45), automargin=True)
)
st.plotly_chart(fig, use_container_width=True)


col1, col2 = st.columns(2, gap='Large')
with col1:
    st.write("### Histogram graph based on Temperature:")
    #st.pyplot(fig9)
    fig, ax = plt.subplots()
    ax.hist(df['temperature_2m'], bins=20)
    st.pyplot(fig)
with col2:
    st.write("### BoxPlot graph based on Temperature in raingn days:")
    fig, ax = plt.subplots(figsize=(15,8))
    sb.boxplot(df['temperature_2m'][df['rain']==1])
    st.pyplot(fig)
    st.write("So as described from above chart, the more rainfalls are between 10 to 17 temperatures value.")
    

