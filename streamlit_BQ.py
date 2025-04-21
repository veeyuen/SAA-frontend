# streamlit_BQ.py
# OCTC Selection for SAA Athletes

import streamlit as st
import pandas as pd
import numpy as np
import datetime
import seaborn as sns
import db_dtypes
import plotly.express as px
import analytics
import re
import gcsfs
from st_files_connection import FilesConnection
from functions import convert_time, process_benchmarks, process_results, map_events, event_date
from google.cloud import storage
from mitosheet.streamlit.v1 import spreadsheet

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


#storage_client = storage.Client(credentials=credentials)

#blobs = storage_client.list_blobs("octc_athletes")

#files = []

#for blob in blobs:
#        files.append(blob)


#st.write("Choose file to open:")
#blob_name = st.multiselect(
#    "Please select the desired file:", 
#    files,)

#bucket = storage_client.bucket("octc_athletes")
#blob = bucket.blob(blob_name)
# with blob.open("r") as f:
#        st.write("READ")

#@st.cache(persist=True)

# Perform query.
# Uses st.cache_data to only rerun when the query changes or after 10 min.
#@st.cache_data(ttl=600)

## Get name variations from GCS bucket


#names = pd.read_csv(file_path, sep=",")


    
### DEFINE SQL QUERIES ###

benchmark_sql = """
SELECT YEAR, EVENT, SUB_EVENT, GENDER, NAME, RESULT, RANK, CATEGORY_EVENT, COMPETITION, STAGE, HEAT
FROM `saa-analytics.benchmarks.saa_benchmarks_prod`
WHERE YEAR='2023' AND COMPETITION='Southeast Asian Games' AND (RANK='3' OR RANK='3.0')
"""

athletes_sql="""
SELECT NAME, RESULT, TEAM, AGE, RANK AS COMPETITION_RANK, DIVISION, EVENT, DISTANCE, EVENT_CLASS, UNIQUE_ID, DOB, NATIONALITY, WIND, CATEGORY_EVENT, GENDER, COMPETITION, YEAR, REGION
FROM `saa-analytics.results.athlete_results_prod` 
WHERE RESULT!='NM' AND RESULT!='-' AND RESULT!='DNS' AND RESULT!='DNF' AND RESULT!='DNQ' AND RESULT!='DQ' AND RESULT IS NOT NULL
"""
all_sql="""
SELECT * FROM `saa-analytics.results.athlete_results_prod`
"""



## Download all data from BQ

df = client.query_and_wait(all_sql).to_dataframe()

df.dropna(how= "all", axis=1, inplace=True)

df = event_date(df)  # call function

start_date = st.date_input("Input start period (dd/mm/yyyy)", format = 'DD/MM/YYYY')
end_date = st.date_input("Input end period (dd/mm/yyy)", format = 'DD/MM/YYYY') 

df['event_date_dt'] = pd.to_datetime(df['event_date'], errors='coerce')

start = np.datetime64(start_date)
end = np.datetime64(end_date)

mask = (df['event_date_dt'] >= start) & (df['event_date_dt'] <= end)
athletes_selected = df.loc[mask]


## Allow public access via mito
final_dfs, code = spreadsheet(athletes_selected)
#st.write(final_dfs)




### EXTRACT LIST OF ATHLETES ###

#athletes = client.query_and_wait(athletes_sql).to_dataframe()

# SELECT YEARS

#selection = client.query_and_wait(all_sql).to_dataframe()

#selection.dropna(how= "all", axis=1, inplace=True)

#year_list = selection['YEAR'].unique().tolist() # get unique list of years
#region_list = df['REGION'].unique().tolist()
#competition_list = df['COMPETITION'].unique().tolist()

#year_selection = st.multiselect(
#    "Please select the desired year(s):",
#    year_list,
#)

#athletes = selection[selection['YEAR'].isin(year_selection)] # filter results based on selected year

## Map relevant events to a standard description ##

athletes_selected['MAPPED_EVENT']=''

map_events(athletes_selected) # call function

#st.write(athletes_selected.shape)

### PROCESS BENCHMARKS ###

benchmarks = client.query_and_wait(benchmark_sql).to_dataframe()

benchmarks=benchmarks[benchmarks['HEAT'].isnull() & benchmarks['SUB_EVENT'].isnull()]  # r

benchmarks.rename(columns = {'RESULT':'BENCHMARK'}, inplace = True)
benchmarks.drop(['YEAR', 'HEAT', 'NAME', 'RANK', 'CATEGORY_EVENT', 'COMPETITION', 'STAGE'], axis=1, inplace=True)

## Convert times in benchmarks to standard format

benchmarks = benchmarks.reset_index(drop=True)

process_benchmarks(benchmarks) # call function

st.write(" ")
"""
## Calculate benchmarks for timed and distance events separately

mask = benchmarks['EVENT'].str.contains(r'jump|throw|Pole|put|Jump|Throw|pole|Put', na=True)


# For distance events

#st.write(benchmarks)

benchmarks.loc[mask, '2%']=benchmarks['Metric']*0.98
benchmarks.loc[mask, '3.5%']=benchmarks['Metric']*0.965
benchmarks.loc[mask, '5%']=benchmarks['Metric']*0.95

# For timed events

benchmarks.loc[~mask, '2%']=benchmarks['Metric']*1.02
benchmarks.loc[~mask, '3.5%']=benchmarks['Metric']*1.035
benchmarks.loc[~mask, '5%']=benchmarks['Metric']*1.05

# Merge benchmarks with df

benchmarks['MAPPED_EVENT']=benchmarks['EVENT'].str.strip()

df = athletes_selected.reset_index().merge(benchmarks.reset_index(), on=['MAPPED_EVENT','GENDER'], how='left')
df['RESULT'] = df['RESULT'].replace(regex=r'–', value=np.nan)
df['RESULT'] = df['RESULT'].replace(regex=r'-', value=np.nan)


## Convert df results into seconds format

st.write(" ")
st.write(" ")

process_results(df) # call fuction

# Fill empty age values

#df["AGE"].fillna(0, inplace=True)
#df['AGE'] = df['AGE'].astype('float')

# Apply OCTC criteria

#octc_df = df.loc[(((df['CATEGORY_EVENT']=='Mid')|(df['CATEGORY_EVENT']=='Sprint')|(df['CATEGORY_EVENT']=='Long')|(df['CATEGORY_EVENT']=='Hurdles')|(df['CATEGORY_EVENT']=='Walk')|(df['CATEGORY_EVENT']=='Relay')|(df['CATEGORY_EVENT']=='Marathon')|(df['CATEGORY_EVENT']=='Steeple')|(df['CATEGORY_EVENT']=='Pentathlon')|(df['CATEGORY_EVENT']=='Heptathlon')|(df['CATEGORY_EVENT']=='Triathlon'))&(df['RESULT_CONV'] <= df['5pc']) & (df['AGE']<40) & ((df['MAPPED_EVENT']!='Marathon')|(df['AGE']<60) & (df['MAPPED_EVENT']=='Marathon')))|(((df['CATEGORY_EVENT']=='Jump')|(df['CATEGORY_EVENT']=='Throw'))&(df['RESULT_CONV'] >= df['5pc']) & (df['AGE']<40) & ((df['MAPPED_EVENT']!='Marathon')|(df['AGE']<60) & (df['MAPPED_EVENT']=='Marathon')))]

df[['2%', '3.5%', '5%', 'RESULT_CONV']] = df[['2%', '3.5%', '5%', 'RESULT_CONV']].apply(pd.to_numeric)

# Measure against 2%, 3.5% and 5% of SEAG 3rd place

mask = df['CATEGORY_EVENT'].str.contains(r'Jump|Throw|jump|throw', na=True)

# For distance events

df.loc[mask, 'Delta2'] = df['RESULT_CONV']-df['2%']
df.loc[mask, 'Delta3.5'] = df['RESULT_CONV']-df['3.5%']
df.loc[mask, 'Delta5'] = df['RESULT_CONV']-df['5%']
df.loc[mask, 'Delta_Benchmark'] = df['RESULT_CONV']-df['Metric']

# For timed events

df.loc[~mask, 'Delta2'] =  df['2%'] - df['RESULT_CONV']
df.loc[~mask, 'Delta3.5'] = df['3.5%'] - df['RESULT_CONV']
df.loc[~mask, 'Delta5'] = df['5%'] - df['RESULT_CONV']
df.loc[~mask, 'Delta_Benchmark'] = df['Metric'] - df['RESULT_CONV']

df=df.loc[df['COMPETITION']!='Southeast Asian Games']

# Create scalar to measure relative performance

df['PERF_SCALAR']=df['Delta5']/df['Metric']*100




# Name corrections
# Read name variations from GCS name lists bucket (Still in beta)


df['NAME'] = df['NAME'].str.replace('\xa0', '', regex=True)
df['NAME'] = df['NAME'].str.replace('[\x00-\x1f\x7f-\x9f]', '', regex=True)
df['NAME'] = df['NAME'].str.replace('\r', '', regex=True)
df['NAME'] = df['NAME'].str.replace('\n', '', regex=True)
df['NAME'] = df['NAME'].str.strip()


# Read csv of name variations from GCS bucket

conn = st.connection('gcs', type=FilesConnection, ttl=600)

names = conn.read("name_variations/name_variations.csv", input_format="csv")


# Iterate over dataframe and replace names

for index, row in names.iterrows():
        
    df['NAME'] = df['NAME'].replace(regex=rf"{row['VARIATION']}", value=f"{row['NAME']}")

# Read list of foreigners from GCS bucket

file_path = "gs://name_lists/List of Foreigners.csv"
#foreigners = pd.read_csv(file_path,
#                 sep=",",
#                 encoding="unicode escape")

#conn = st.connection("gsheets", type=GSheetsConnection, worksheet="Sheet2")
#foreigners = conn.read()

#st.write(foreigners)


# Process list of foreign names and their variations

df_local_teams = df[(df['TEAM']!='Malaysia')&(df['TEAM']!='THAILAND')&(df['TEAM']!='China') 
                       &(df['TEAM']!='South Korea')&(df['TEAM']!='Laos') 
                       &(df['TEAM']!='Philippines')&(df['TEAM']!='Piboonbumpen Thailand') 
                       &(df['TEAM']!='Chinese Taipei')&(df['TEAM']!='Gurkha Contingent') 
                       &(df['TEAM']!='Australia')&(df['TEAM']!='Piboonbumpen Thailand') 
                       &(df['TEAM']!='Hong Kong')&(df['TEAM']!='PERAK')&(df['TEAM']!='Sri Lanka') 
                       &(df['TEAM']!='Indonesia')&(df['TEAM']!='THAILAND')&(df['TEAM']!='MALAYSIA') 
                       &(df['TEAM']!='PHILIPPINES') & (df['TEAM']!='SOUTH KOREA')&(df['TEAM']!='Waseda') 
                       &(df['TEAM']!='LAOS')&(df['TEAM']!='CHINESE TAIPEI')
                       &(df['TEAM']!='INDIA')&(df['TEAM']!='Hong Kong, China')&(df['TEAM']!='AIC JAPAN')] 

top_performers_clean = df_local_teams.sort_values(['MAPPED_EVENT', 'NAME','PERF_SCALAR'],ascending=False).groupby(['MAPPED_EVENT', 'NAME']).head(1)


top_performers_clean.reset_index(inplace=True)



# Create performance tier column

top_performers_clean['TIER'] = np.where((top_performers_clean['Delta_Benchmark']>=0), 'Tier 1',    
                                np.where(((top_performers_clean['Delta_Benchmark']<0) & (top_performers_clean['Delta2']>=0)), 'Tier2',
                                np.where(((top_performers_clean['Delta2']<0) & (top_performers_clean['Delta3.5']>=0)), 'Tier3', 
                                np.where(((top_performers_clean['Delta3.5']<0) & (top_performers_clean['Delta5']>=0)), 'Tier4', ' '))))


# Drop rows without a corresponding benchmark

final_df = top_performers_clean[top_performers_clean['BENCHMARK'].notna()]

# Show resulting OCTC dataframe

st.write(" ")
st.write(" ")
st.write(" ")



st.write("LIST OF OCTC SELECTION ATHLETES:")

st.write(final_df)


# Process custom threshold benchmark

benchmarks['custom']=''

input = st.number_input("Input desired benchmark threshold (%):")

mask = benchmarks['EVENT'].str.contains(r'jump|throw|Pole|put', na=True)

# For distance events

benchmarks.loc[mask, 'custom']=benchmarks['Metric']*((100-input)/100)

# For timed events

benchmarks.loc[~mask, 'custom']=benchmarks['Metric']*((100+input)/100)


temp_df = athletes.reset_index().merge(benchmarks.reset_index(), on=['MAPPED_EVENT','GENDER'], how='left')
temp_df['RESULT'] = athletes['RESULT'].replace(regex=r'–', value=np.nan)

for i in range(len(temp_df)):
     
    rowIndex = temp_df.index[i]

    input_string=temp_df.iloc[rowIndex,5]    
    
    metric=temp_df.iloc[rowIndex,2]
    
    if metric=='—' or metric=='DQ' or metric=='SCR' or metric=='FS' or metric=='DNQ' or metric==' DNS' or metric=='NH':
        continue 
        
    out = convert_time(i, input_string, metric)
         
    temp_df.loc[rowIndex, 'RESULT_CONV'] = out

temp_df["AGE"].fillna(0, inplace=True)
temp_df['AGE'] = temp_df['AGE'].astype('float')



# Create new df for custom benchmarks

custom_df = temp_df.loc[(((temp_df['CATEGORY_EVENT']=='Mid')|(temp_df['CATEGORY_EVENT']=='Sprint')|(temp_df['CATEGORY_EVENT']=='Long')|(temp_df['CATEGORY_EVENT']=='Hurdles')|(temp_df['CATEGORY_EVENT']=='Walk')|(temp_df['CATEGORY_EVENT']=='Relay')|(temp_df['CATEGORY_EVENT']=='Marathon')|(temp_df['CATEGORY_EVENT']=='Steeple')|(temp_df['CATEGORY_EVENT']=='Pentathlon')|(temp_df['CATEGORY_EVENT']=='Heptathlon')|(temp_df['CATEGORY_EVENT']=='Triathlon'))&(temp_df['RESULT_CONV'] <= temp_df['custom']) & (temp_df['AGE']<40) & ((temp_df['MAPPED_EVENT']!='Marathon')|(temp_df['AGE']<60) & (temp_df['MAPPED_EVENT']=='Marathon')))|(((temp_df['CATEGORY_EVENT']=='Jump')|(temp_df['CATEGORY_EVENT']=='Throw'))&(temp_df['RESULT_CONV'] >= temp_df['custom']) & (temp_df['AGE']<40) & ((temp_df['MAPPED_EVENT']!='Marathon')|(temp_df['AGE']<60) & (temp_df['MAPPED_EVENT']=='Marathon')))]

custom_df=custom_df.loc[custom_df['COMPETITION']!='SEA Games']


st.write(custom_df)


# Measure against 2%, 3.5% and 5% of SEAG 3rd place

mask = custom_df['CATEGORY_EVENT'].str.contains(r'Jump|Throw', na=True) 

# For distance events
"""

