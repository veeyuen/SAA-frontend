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
from functions import convert_time, process_benchmarks, process_results, map_international_events, event_date, clean_columns
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


## Create BigQuery API client ##

credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"]
)

client = bigquery.Client(credentials=credentials)

## Initialize Session State ##

if 'option' not in st.session_state:
        st.session_state.option = None



#storage_client = storage.Client(credentials=credentials)
    
### DEFINE SQL QUERIES ###

seag_benchmark_sql = """
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

## Read all performance benchmarks csv from GCS bucket and process##

conn = st.connection('gcs', type=FilesConnection, ttl=600)
benchmarks = conn.read("competition_benchmarks/All_Benchmarks_Processed.csv", input_format="csv")

#process_benchmarks(benchmarks)

#st.write(benchmarks)


## Download all athlete data from BQ

data = client.query_and_wait(all_sql).to_dataframe()

data.dropna(how= "all", axis=1, inplace=True)

data = event_date(data)  # call function

start_date = st.date_input("Input Start Period (dd/mm/yyyy)", format = 'DD/MM/YYYY')
end_date = st.date_input("Input End Period (dd/mm/yyy)", format = 'DD/MM/YYYY') 

data['event_date_dt'] = pd.to_datetime(data['event_date'], errors='coerce')

start = np.datetime64(start_date)
end = np.datetime64(end_date)

mask = (data['event_date_dt'] >= start) & (data['event_date_dt'] <= end)
athletes_selected = data.loc[mask]


## Allow public access via mito

#final_dfs, code = spreadsheet(athletes_selected)

benchmark_option = st.selectbox(
    "Please Select Performance Benchmark (Select 'None' to Access All Records in Database)",
    ("None", "2023 SEAG Bronze", "26th Asian Athletics", "2025 Taiwan Open"),
)

if benchmark_option == 'None':

    final_dfs, code = spreadsheet(athletes_selected)

    benchmark = pd.DataFrame()

elif benchmark_option == '2023 SEAG Bronze':

    benchmark = benchmarks[benchmarks['BENCHMARK']== '2023 SEAG Bronze']

    option='option1'

elif benchmark_option == '26th Asian Athletics':

    benchmark = benchmarks[benchmarks['BENCHMARK']== '26th Asian Athletics']

    option='option2'

elif benchmark_option == '2025 Taiwan Open':

    benchmark = benchmarks[benchmarks['BENCHMARK']== '2025 Taiwan Open']

    option='option3'


## Map relevant events to a standard description ##

athletes_selected['MAPPED_EVENT']=''

map_international_events(athletes_selected) # call function


### PROCESS BENCHMARKS###

## Convert benchmarks results to float64 compatible format ##

#st.write(benchmark)


## Prepare to merge benchmarks with athlete df ##

#benchmarks['MAPPED_EVENT']=benchmarks['EVENT'].str.strip()  # create MAPPED_EVENT column from EVENT column

#st.write(athletes_selected.columns)

st.write(option)
st.write(st.session_state.option)

if benchmark_option != 'None' and st.session_state.option is None:

    st.write('HERE')

    df = pd.merge(
        left=athletes_selected, 
        right=benchmark,
        how='left',
        left_on=['MAPPED_EVENT', 'GENDER'],
        right_on=['EVENT', 'GENDER'],
    )                   
    
    
    clean_columns(df) # clean benchmarks of hidden characters, spaces etc. to ensure proper merging

    
   # st.write(df.columns)


    df['RESULT_x'] = df['RESULT_x'].replace(regex=r'–', value=np.nan)

    process_results(df) # call fuction

    st.session_state.option = df

## Convert athlete results into float64 compatible format

    

    st.write(df.columns)

## Create scalar to measure relative performance - distance events are reversed from timed events ##

    df['PERF_SCALAR']=df['Delta5']/df['Metric']*100

# Name corrections
# Read name variations from GCS name lists bucket (Still in beta)

#clean_columns(df) 

# Read csv of name variations from GCS bucket

#conn = st.connection('gcs', type=FilesConnection, ttl=600)
#names = conn.read("name_variations/name_variations.csv", input_format="csv")

# Iterate over dataframe and replace names

#for index, row in names.iterrows():
        
#    df['NAME'] = df['NAME'].replace(regex=rf"{row['VARIATION']}", value=f"{row['NAME']}")

# Read list of foreigners from GCS bucket

#conn = st.connection('gcs', type=FilesConnection, ttl=600)
#foreigners = conn.read("name_lists/List of Foreigners.csv", encoding="unicode escape", input_format="csv")

# Remove foreigners

#foreigners['V1'] = foreigners['LAST_NAME']+' '+foreigners['FIRST_NAME']
#foreigners['V2'] = foreigners['FIRST_NAME']+' '+foreigners['LAST_NAME']
#foreigners['V3'] = foreigners['LAST_NAME']+', '+foreigners['FIRST_NAME']
#foreigners['V4'] = foreigners['FIRST_NAME']+' '+foreigners['LAST_NAME']

#for1 = foreigners['V1'].dropna().tolist()
#for2 = foreigners['V2'].dropna().tolist()
#for3 = foreigners['V3'].dropna().tolist()
#for4 = foreigners['V4'].dropna().tolist()

#foreign_list = for1+for2+for3+for4 

#foreign_list_casefold=[s.casefold() for s in foreign_list]

#exclusions = foreign_list_casefold

#no_foreigners_list = df_select.loc[~df['NAME'].str.casefold().isin(exclusions)]  # ~ means NOT IN. DROP spex carded athletes

# Choose the best result for each event participated by every athlete

#top_performers = no_foreigners_list.sort_values(['MAPPED_EVENT', 'NAME','PERF_SCALAR'],ascending=False).groupby(['MAPPED_EVENT', 'NAME']).head(1)

#st.write(foreigners)


# Process list of foreign names and their variations

#df_local_teams = df[(df['TEAM']!='Malaysia')&(df['TEAM']!='THAILAND')&(df['TEAM']!='China') 
#                       &(df['TEAM']!='South Korea')&(df['TEAM']!='Laos') 
#                       &(df['TEAM']!='Philippines')&(df['TEAM']!='Piboonbumpen Thailand') 
#                       &(df['TEAM']!='Chinese Taipei')&(df['TEAM']!='Gurkha Contingent') 
#                       &(df['TEAM']!='Australia')&(df['TEAM']!='Piboonbumpen Thailand') 
#                       &(df['TEAM']!='Hong Kong')&(df['TEAM']!='PERAK')&(df['TEAM']!='Sri Lanka') 
#                       &(df['TEAM']!='Indonesia')&(df['TEAM']!='THAILAND')&(df['TEAM']!='MALAYSIA') 
#                       &(df['TEAM']!='PHILIPPINES') & (df['TEAM']!='SOUTH KOREA')&(df['TEAM']!='Waseda') 
#                       &(df['TEAM']!='LAOS')&(df['TEAM']!='CHINESE TAIPEI')
#                       &(df['TEAM']!='INDIA')&(df['TEAM']!='Hong Kong, China')&(df['TEAM']!='AIC JAPAN')] 

#top_performers_clean = df_local_teams.sort_values(['MAPPED_EVENT', 'NAME','PERF_SCALAR'],ascending=False).groupby(['MAPPED_EVENT', 'NAME']).head(1)


#top_performers_clean.reset_index(inplace=True)



# Create performance tier column

#top_performers_clean['TIER'] = np.where((top_performers_clean['Delta_Benchmark']>=0), 'Tier 1',    
#                                np.where(((top_performers_clean['Delta_Benchmark']<0) & (top_performers_clean['Delta2']>=0)), 'Tier2',
#                                np.where(((top_performers_clean['Delta2']<0) & (top_performers_clean['Delta3.5']>=0)), 'Tier3', 
#                                np.where(((top_performers_clean['Delta3.5']<0) & (top_performers_clean['Delta5']>=0)), 'Tier4', ' '))))


# Drop rows without a corresponding benchmark

#final_df = top_performers_clean[top_performers_clean['BENCHMARK'].notna()]

# Show resulting OCTC dataframe

#st.write(" ")
#st.write(" ")
#st.write(" ")



#st.write("LIST OF OCTC SELECTION ATHLETES:")

#st.write(final_df)


# Process custom threshold benchmark

#benchmarks['custom']=''

#input = st.number_input("Input desired benchmark threshold (%):")

#mask = benchmarks['EVENT'].str.contains(r'jump|throw|Pole|put', na=True)

# For distance events

#benchmarks.loc[mask, 'custom']=benchmarks['Metric']*((100-input)/100)

# For timed events

#benchmarks.loc[~mask, 'custom']=benchmarks['Metric']*((100+input)/100)


#temp_df = athletes.reset_index().merge(benchmarks.reset_index(), on=['MAPPED_EVENT','GENDER'], how='left')
#temp_df['RESULT'] = athletes['RESULT'].replace(regex=r'–', value=np.nan)

#for i in range(len(temp_df)):
     
#    rowIndex = temp_df.index[i]

#    input_string=temp_df.iloc[rowIndex,5]    
    
 #   metric=temp_df.iloc[rowIndex,2]
    
 #   if metric=='—' or metric=='DQ' or metric=='SCR' or metric=='FS' or metric=='DNQ' or metric==' DNS' or metric=='NH':
 #       continue 
        
 #   out = convert_time(i, input_string, metric)
         
  #  temp_df.loc[rowIndex, 'RESULT_CONV'] = out

#temp_df["AGE"].fillna(0, inplace=True)
#temp_df['AGE'] = temp_df['AGE'].astype('float')



# Create new df for custom benchmarks

#custom_df = temp_df.loc[(((temp_df['CATEGORY_EVENT']=='Mid')|(temp_df['CATEGORY_EVENT']=='Sprint')|(temp_df['CATEGORY_EVENT']=='Long')|(temp_df['CATEGORY_EVENT']=='Hurdles')|(temp_df['CATEGORY_EVENT']=='Walk')|(temp_df['CATEGORY_EVENT']=='Relay')|(temp_df['CATEGORY_EVENT']=='Marathon')|(temp_df['CATEGORY_EVENT']=='Steeple')|(temp_df['CATEGORY_EVENT']=='Pentathlon')|(temp_df['CATEGORY_EVENT']=='Heptathlon')|(temp_df['CATEGORY_EVENT']=='Triathlon'))&(temp_df['RESULT_CONV'] <= temp_df['custom']) & (temp_df['AGE']<40) & ((temp_df['MAPPED_EVENT']!='Marathon')|(temp_df['AGE']<60) & (temp_df['MAPPED_EVENT']=='Marathon')))|(((temp_df['CATEGORY_EVENT']=='Jump')|(temp_df['CATEGORY_EVENT']=='Throw'))&(temp_df['RESULT_CONV'] >= temp_df['custom']) & (temp_df['AGE']<40) & ((temp_df['MAPPED_EVENT']!='Marathon')|(temp_df['AGE']<60) & (temp_df['MAPPED_EVENT']=='Marathon')))]

#custom_df=custom_df.loc[custom_df['COMPETITION']!='SEA Games']


#st.write(custom_df)


# Measure against 2%, 3.5% and 5% of SEAG 3rd place

#mask = custom_df['CATEGORY_EVENT'].str.contains(r'Jump|Throw', na=True) 

# For distance events


