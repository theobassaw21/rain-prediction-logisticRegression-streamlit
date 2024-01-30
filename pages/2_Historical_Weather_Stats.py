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

st.sidebar.write("""
Helpful Menu:
- [Top](#statistis-on-weather-historical-data-for-athens)
- [Temp mean values & Increase Percentage](#means-values-increase-percentage-temperature-for-every-five-years)
- [Total Days & Raining Days](#total-days-raining-days-for-every-temperature-category)
- [Statistics when it rains](#statistics-when-it-rains)
- [Precipitation per year](#display-the-sum-of-precipitation-per-year)
- [Precipitation per month](#display-the-sum-of-precipitation-per-month-for-every-year)
- [Precipitation per hour](#display-the-sum-of-precipitation-per-hour)

 """)

@st.cache_data(persist="disk")
def read_csv():
    df = pd.read_csv('historical_data/hourly_till_now.csv', skiprows=range(1, 3))
    #df = df.drop(['Unnamed: 0'], axis=1)
    df["date"] = pd.to_datetime(df["date"])
    df['hour'] = df['date'].dt.hour

    df['day'] = df['date'].dt.strftime('%d')
    df['month'] = df['date'].dt.strftime('%m')
    df['year'] = df['date'].dt.strftime('%Y')
    df['year'] = pd.to_datetime(df["year"])
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


df, len_df= read_csv()
start_date = pd.Timestamp(st.sidebar.date_input("Start date", min_value=df['date'].min().date(), max_value=df['date'].max().date(), value= df['date'].min().date()))
end_date = pd.Timestamp(st.sidebar.date_input("End date", min_value=df['date'].min().date(), max_value=df['date'].max().date(), value= df['date'].max().date()))
df = df[(df['date'] > start_date) & (df['date'] <= end_date)]

st.write("## Statistis on Weather Historical Data for Athens:")
#st.write("#### Statistis on Weather Historical Data for Athens in time period:", start_date, " to", end_date)
st.write("#### Weather historical data: \n Period:", start_date, " to", end_date)

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

st.write("---")
st.write("#### Means values & Increase percentage temperature for every five years\n - Period:", start_date, " to", end_date)
st.write("#")
col1, col2 = st.columns([1,5])

with col1:
    mean_temp_2004_2008 = round(df['temperature_2m'][ (df['year'] >='2004') & (df['year'] <='2008') ].mean(),4)
    #st.write("Mean Temp 2004 to 2008")
    st.metric(label="Mean Temp 2004 to 2008:", value=f"{mean_temp_2004_2008} °C")
    mean_temp_2009_2013 = round(df['temperature_2m'][ (df['year'] >='2009') & (df['year'] <='2013') ].mean(),4)
    #st.write("Mean Temp 2009 to 2013")
    #st.write("Increase percentage: ", round(mean_temp_2009_2013 / mean_temp_2004_2008,4)  )
    st.metric(label="Mean Temp 2009 to 2013", value=f"{mean_temp_2009_2013} °C", delta=f"{round(mean_temp_2009_2013 / mean_temp_2004_2008,4)} °C")
    mean_temp_2014_2018 = round(df['temperature_2m'][ (df['year'] >='2014') & (df['year'] <='2018') ].mean(),4)
    #st.write("Mean Temp 2014 to 2018")
    #st.write("Increase percentage: ", round(mean_temp_2014_2018 / mean_temp_2009_2013,4)  )
    st.metric(label="Mean Temp 2018 to 2018:", value=f"{mean_temp_2014_2018} °C", delta=f"{round(mean_temp_2014_2018 / mean_temp_2009_2013,4)} °C")
    mean_temp_2019_2023 = round(df['temperature_2m'][ (df['year'] >='2019') & (df['year'] <='2023') ].mean(),4)
    #st.write("Mean Temp 2019 to 2023")
    #st.write("Increase percentage: ", round(mean_temp_2019_2023 / mean_temp_2014_2018,4)  )
    st.metric(label="Mean Temp 2019 to 2023", value=f"{mean_temp_2019_2023} °C", delta=f"{round(mean_temp_2019_2023 / mean_temp_2014_2018,4)} °C")

# with col2:
    
# with col3:
    
with col2:
#with col5:
    data = [mean_temp_2004_2008, mean_temp_2009_2013, mean_temp_2014_2018, mean_temp_2019_2023]
    df_tmp = pd.DataFrame(data)
    fig = px.line(df_tmp,width=900, height=500,
    labels={
                        "df_tmp.index" ,
                        "df_tmp.values",
                        
                    })
    fig.update_layout(
        title=dict(text="Percentage of change of mean temperature per five years", font=dict(size=25), automargin=True)
    )
    st.plotly_chart(fig, use_container_width=True)



#df_tmp1 = df['temperature_2m'].groupby(df['year']).mean()

st.write("---")

co1, co2 = st.columns(2)
with co1:
    st.write("#### Created five categories for temperature to summarize the values.")

    st.write("""
    - category 0 : where temperature is under 0 \n 
    - category 1 : where temperature is between 0 and 10 \n
    - category 3 : where temperature is between 20 and 30 \n
    - category 4 : where temperature is between 30 and 40\n
    - category 4 : where temperature is between 30 and 40\n
    - category 5 : where temperature is over 40 \n
            """)
    # col1, col2, col3 = st.columns(3)
    # with col1:
    #     st.write("""\n
    #         - category 0 : where temperature is under 0 \n 
    #         - category 1 : where temperature is between 0 and 10 \n
    #         """)
    # with col2:
    #     st.write("""\n
    #         - category 3 : where temperature is between 20 and 30 \n
    #         - category 4 : where temperature is between 30 and 40\n
    #         """)
    # with col3:
    #     st.write("""\n
    #         - category 4 : where temperature is between 30 and 40\n
    #         - category 5 : where temperature is over 40 \n""")
with co2:
    st.write("#### Counts for each category of temperature:")
    fig = px.pie(df, names='temperature_2m_category')
    fig.update_traces(hoverinfo='label+percent', textinfo='label+percent+value')
    fig.update_layout(
    
    )
    st.plotly_chart(fig, use_container_width=True)  


st.write("#### Total days & Raining days for every temperature category\n - Period:", start_date, " to", end_date)
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

st.write("---")


raining_days_count_per_month = (df['month'][df['rain']==1].value_counts() / 240)
st.write("##### Average number of raining days per month\n - Period:", start_date, " to", end_date)


fig = px.bar(df, x=raining_days_count_per_month.index , y= raining_days_count_per_month.values , width=900, height=500)
fig.update_layout(margin=dict(l=20, r=20, t=50, b=20))
fig.update_xaxes(rangemode='tozero', showgrid=False)
fig.update_yaxes(rangemode='tozero', showgrid=True)
st.plotly_chart(fig, use_container_width=True)

# col1 , col2 = st.columns(2, gap='Large')
# with col1:
#     st.write("##### Counts for each category of temperature:\n - Period:", start_date, " to", end_date)
#     fig = px.pie(df, names='temperature_2m_category')
#     fig.update_traces(hoverinfo='label+percent', textinfo='label+percent+value')
#     fig.update_layout(
    
#     )
#     st.plotly_chart(fig, use_container_width=True)  

# with col2:
#     raining_days_count_per_month = (df['month'][df['rain']==1].value_counts() / 240)
#     st.write("##### Average number of raining days per month\n - Period:", start_date, " to", end_date)

    
#     fig = px.bar(df, x=raining_days_count_per_month.index , y= raining_days_count_per_month.values , width=900, height=500)
#     fig.update_layout(margin=dict(l=20, r=20, t=50, b=20))
#     fig.update_xaxes(rangemode='tozero', showgrid=False)
#     fig.update_yaxes(rangemode='tozero', showgrid=True)
#     st.plotly_chart(fig, use_container_width=True)

st.write("#")
st.write("##### Display data based on time duration - Period:", start_date, " to", end_date)
st.write("#### Statistics when it rains:")


col1, col2, col3, col4 = st.columns(4)
with col1:
    st.write("-Top 5 Months with most raining days:")
    top_months = raining_days_count_per_month.nlargest(n=5)
    st.write(top_months)
with col2:
    st.write("-Top hours list, that usually rain:")
    top_raining_hours = df['hour'][df['rain']==1].value_counts().nlargest(n=5,)
    st.write(top_raining_hours)
with col3:
    st.write("-Top hours list, that usually doesn't rain:")
    top_raining_hours = df['hour'][df['rain']==0].value_counts().nlargest(n=5,)
    st.write(top_raining_hours)
with col4:
    st.write("- Dates of most hourly precipitation:")
    most_hourly_precipitation = df[['date', 'precipitation']].nlargest(n=5, columns=['precipitation'])
    st.write(most_hourly_precipitation)


##########------Precipitation per year--------------##############
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
    title=dict(text="Precipitation per year", font=dict(size=45), automargin=True)
)
st.plotly_chart(fig, use_container_width=True)



##########------Precipitation per month--------------##############
st.write("---")
st.write("##### Display the sum of precipitation per month for every year:\n - Period:", start_date, " to", end_date)
choose_year_options = df['year'].dt.strftime('%Y').unique()
choose_year = st.selectbox(
    label = 'Choose year to display precipitation per month',index=len(choose_year_options)-2 if len(choose_year_options)>=2 else  len(choose_year_options)-1, options = choose_year_options)
st.write("#")
df_tmp = df['precipitation'][df['year'] == choose_year].groupby(df.month).sum()
fig = px.bar(df_tmp, y="precipitation" , width=900, height=500,
labels={
                     "df['precipitation'].groupby(df.month).sum().index": "Sepal Length (cm)",
                     "df['precipitation'].groupby(df.month).sum().values": "Sepal Width (cm)",
                     
                 })
fig.update_layout(
    title=dict(text=f"Precipitation per month for year: {choose_year}", font=dict(size=45), automargin=True)
)
st.plotly_chart(fig, use_container_width=True)



##########------Precipitation per hour--------------##############

st.write("---")
st.write("##### Display the sum of precipitation per hour:\n - Period:", start_date, " to", end_date)
st.write("#")

df_tmp = df['precipitation'].groupby(df.hour).sum()
fig = px.bar(df_tmp, y="precipitation" , width=900, height=500,
labels={
                     "df['precipitation'].groupby(df.year).sum().index": "Sepal Length (cm)",
                     "df['precipitation'].groupby(df.year).sum().values": "Sepal Width (cm)",
                     
                 })
fig.update_layout(
    title=dict(text="Precipitation per hour", font=dict(size=45), automargin=True)
)
st.plotly_chart(fig, use_container_width=True)


col1, col2 = st.columns(2, gap='Large')
with col1:
    st.write("### Histogram graph based on Temperature:")
    #st.pyplot(fig9)
    fig, ax = plt.subplots()
    ax.hist(df['temperature_2m'], bins=20)
    st.pyplot(fig)

    # st.write("### Histogram graph based on precipitation per hour:")
    # #st.pyplot(fig9)
    # fig, ax = plt.subplots()
    # ax.hist(df['precipitation'].groupby(df.month).sum(), bins=20)
    # st.pyplot(fig)
with col2:
    st.write("### BoxPlot graph based on Temperature in raingn days:")
    fig, ax = plt.subplots(figsize=(15,8))
    sb.boxplot(df['temperature_2m'][df['rain']==1])
    st.pyplot(fig)
    st.write("So as described from above chart, the more rainfalls are between 10 to 17 temperatures value.")
   
