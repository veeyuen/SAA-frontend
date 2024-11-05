# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import datetime
import seaborn as sns
import db_dtypes
import plotly.express as px
#from mitosheet.streamlit.v1 import spreadsheet
import analytics



from matplotlib import pyplot as plt

from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)

from google.oauth2 import service_account
from google.cloud import bigquery


# Create API client.
credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"]
)
client = bigquery.Client(credentials=credentials)


#@st.cache(persist=True)

# Perform query.
# Uses st.cache_data to only rerun when the query changes or after 10 min.
@st.cache_data(ttl=600)
def run_query(query):
    query_job = client.query(query)
    rows_raw = query_job.result()
    # Convert to list of dicts. Required for st.cache_data to hash the return value.
    rows = [dict(row) for row in rows_raw]
    return rows

#rows = run_query("SELECT word FROM `bigquery-public-data.samples.shakespeare` LIMIT 10")

#rows = run_query("SELECT * FROM `saa-analytics.results.saa_full` LIMIT 10")


# Print results.
#st.write("Writing rows from table:")
#for row in rows:
#    st.write("✍️ " + row['word'])
#    st.write(row)

df = client.query_and_wait("""SELECT * FROM `saa-analytics.results.saa_full`""").to_dataframe()

df.dropna(how= "all", axis=1, inplace=True)

year_list=df['DATE'].unique().tolist() # get unique list of source events

event_options = st.multiselect(
    "Please select the desired year(s):",
    year_list,
)

st.write("You selected:", event_options)

df_filtered = df[df['DATE'].isin(event_options)]
#final_dfs, code = spreadsheet(df)

#st.write(final_dfs)

#st.code(code)

st.write(df_filtered)



# Extract year and month

#data['year'] = pd.DatetimeIndex(data['Date']).year
#data['month'] = pd.DatetimeIndex(data['Date']).month

# Filter dataframe


#events = data['Event'].drop_duplicates()
#event_choice = st.sidebar.selectbox('Select the event:', events)
#dates = data['year'].loc[data["Event"] == event_choice].drop_duplicates()


#start_year = st.sidebar.selectbox('Start Year', dates)
#end_year = st.sidebar.selectbox('End Year', dates)

#mask = ((data['year'] >= start_year) & (data['year'] <= end_year) & (data['Event']==event_choice))

#filter=data.loc[mask]

#st.dataframe(filter)


# Plot using Seaborn

#metrics = filter['Metric']

#fig, ax = plt.subplots()
#plt.style.use("dark_background")

#plt.title("Distribution of Times/Distances")
#ax = sns.histplot(data=filter, x='Metric', kde=True, color = "#b80606")

#ax = plt.hist(metrics, bins=7)

#st.pyplot(fig)


# Print stats summary

#summary = metrics.describe()
#st.write(summary)

#col1, col2, col3, col4 = st.columns(4)
#col1.metric("No. Records", value=int(summary[0]))
#col2.metric("Mean", value=summary[1].round(2))
#col3.metric("Standard Dev.", value=summary[2].round(2))
#col4.metric("Min Value", value=summary[3])

#col1, col2, col3, col4 = st.columns(4)
#col1.metric("25 percentile", value=summary[4].round(2))
#col2.metric("50 percentile", value=summary[5].round(2))
#col3.metric("75 percentile", value=summary[6].round(2))
#col4.metric("Max Value", value=summary[7])
