# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import datetime
import seaborn as sns

from matplotlib import pyplot as plt

from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)

from google.oauth2 import service_account
from google.cloud import storage


#bucket_name = "singapore_athletics_association"
#file_path = "consolidated.csv"

## Data preprocess and cleaning
def preprocess(i, string, metric):

    global OP

    l=['discus', 'throw', 'jump', 'vault', 'shot']

    string=string.lower()


    if any(s in string for s in l)==True:

        OP=float(str(metric))



    else:

        searchstring = ":"
        searchstring2 = "."
        substring=str(metric)
        count = substring.count(searchstring)
        count2 = substring.count(searchstring2)

        if count==0:
            OP=float(substring)



        elif (type(metric)==datetime.time or type(metric)==datetime.datetime):

            time=str(metric)
            h, m ,s = time.split(':')
            OP = float(datetime.timedelta(hours=int(h),minutes=int(m),seconds=float(s)).total_seconds())


        elif (count==1 and count2==1):

            m,s = metric.split(':')
            OP = float(datetime.timedelta(minutes=int(m),seconds=float(s)).total_seconds())

        elif (count==1 and count2==2):

            metric = metric.replace(".", ":", 1)

            h,m,s = metric.split(':')
            OP = float(datetime.timedelta(hours=int(h),minutes=int(m),seconds=float(s)).total_seconds())


        elif (count==2 and count2==0):

            h,m,s = metric.split(':')
            OP = float(datetime.timedelta(hours=int(h),minutes=int(m),seconds=float(s)).total_seconds())


    return OP

# Clean each row of input file

def clean(data):

    for i in range(len(data)):

        rowIndex = data.index[i]

        input_string=data.iloc[rowIndex,1]
        metric=data.iloc[rowIndex,5]

        processed_output = preprocess(i, input_string, metric)

        data.loc[rowIndex, 'Metric'] = processed_output

    return data




# Create API client
credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"]
)
client = storage.Client(credentials=credentials)


URL = ("https://storage.googleapis.com/singapore_athletics_association/consolidated.csv")

@st.cache(persist=True)

def load_data():

    client = storage.Client()
    data = pd.read_csv(URL, usecols = ['Date','Event', 'Name', 'Age', 'Team', 'Result', 'm/s', 'Competition',
              'Year D.O.B.', 'Info, if any', 'Metric'])
    return data

data = load_data()


# Filter dataframe


data['year'] = pd.DatetimeIndex(data['Date']).year

events = data['Event'].drop_duplicates()
event_choice = st.sidebar.selectbox('Select the event:', events)
dates = data["Date"].loc[data["Event"] == event_choice]


start_date = st.sidebar.selectbox('Start Date', dates)
end_date = st.sidebar.selectbox('End Date', dates)

mask = ((data['Date'] > start_date) & (data['Date'] <= end_date) & (data['Event']==event_choice))

filter=data.loc[mask]

st.dataframe(filter)


# Plot using Seaborn

metrics = filter['Metric']

fig, ax = plt.subplots()
plt.style.use("dark_background")

plt.title("Distribution of Times/Distances")
ax = sns.histplot(data=filter, x='Metric', kde=True, color = "#b80606")

#ax = plt.hist(metrics, bins=7)

st.pyplot(fig)





# Print stats summary

summary = metrics.describe()
#st.write(summary)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Count", value=int(summary[0]))
col2.metric("Mean", value=summary[1].round(2))
col3.metric("St.dev", value=summary[2].round(2))
col4.metric("Min", value=summary[3])

col1, col2, col3, col4 = st.columns(4)
col1.metric("25%", value=summary[4].round(2))
col2.metric("50%", value=summary[5].round(2))
col3.metric("75%", value=summary[6].round(2))
col4.metric("Max", value=summary[7])



## Upload CSV

def upload_csv(df):

    credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"]
)
    client = storage.Client(credentials=credentials)

    bucket = client.get_bucket('singapore_athletics_association')

    df.to_csv()
    bucket.blob('consolidated.csv').upload_from_string(df.to_csv(), 'text/csv')

    return


with st.sidebar:
    uploaded_file = st.file_uploader("Upload new records via CSV file", accept_multiple_files=False)

if uploaded_file is not None:


    try:

        df_new=pd.read_csv(uploaded_file)

        df_processed=clean(df_new)
        st.dataframe(df_processed)

        # Merge newly created df with previous df

        frames=[df_processed, data]

        upload_df = pd.concat(frames)
        upload_df = upload_df.reset_index(drop=True)

        upload_csv(upload_df)

        st.write("Data uploaded into Google Cloud Storage Bucket")


    except:

        st.warning("Error encountered loading data file. Please check column positions and data formats.")
