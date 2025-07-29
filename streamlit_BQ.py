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
import pytz
from st_files_connection import FilesConnection
from functions import convert_time, process_results, map_international_events, clean_columns
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

#credentials = service_account.Credentials.from_service_account_info(
#    st.secrets["gcp_service_account"]
#)

#client = bigquery.Client(credentials=credentials)

@st.cache_resource
def get_bq_client():
    credentials = service_account.Credentials.from_service_account_info(
        st.secrets["gcp_service_account"]
    )
    return bigquery.Client(credentials=credentials)
client = get_bq_client()


## Read csv file containing list of foreigners ##
@st.cache_data(ttl=500)
def fetch_foreigners():
    conn = st.connection('gcs', type=FilesConnection, ttl=600)
    foreigners = conn.read("name_lists/List of Foreigners.csv", encoding="utf-8", input_format="csv")
    return foreigners
foreigners = fetch_foreigners()  # get list of foreigners

@st.cache_data(ttl=500)
def name_variations():
    conn = st.connection('gcs', type=FilesConnection, ttl=600)
    names = conn.read("name_variations/name_variations.csv", input_format="csv")
    names = clean_columns(names)  # clean name list of special characters, white spaces etc.
    return names
names = name_variations()


# Create list of foreigners 

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


foreigners['V1'] = foreigners['LAST_NAME']+' '+foreigners['FIRST_NAME']
foreigners['V2'] = foreigners['FIRST_NAME']+' '+foreigners['LAST_NAME']
foreigners['V3'] = foreigners['LAST_NAME']+', '+foreigners['FIRST_NAME']
foreigners['V4'] = foreigners['FIRST_NAME']+' '+foreigners['LAST_NAME']
foreign_list = pd.concat([
    foreigners['V1'], foreigners['V2'],
    foreigners['V3'], foreigners['V4']
]).dropna().str.casefold().unique().tolist()
exclusions = set(foreign_list)

### DEFINE SQL QUERIES ###

athletes_sql="""
SELECT NAME, RESULT, TEAM, AGE, RANK AS COMPETITION_RANK, STAGE, DICT_RESULTS, SOURCE, REMARKS, SUB_EVENT,  DIVISION, EVENT, DATE, DISTANCE, EVENT_CLASS, UNIQUE_ID, DOB, NATIONALITY, WIND, CATEGORY_EVENT, GENDER, COMPETITION, YEAR, REGION
FROM `saa-analytics.results.PRODUCTION` 
WHERE RESULT NOT IN ('NM', '-', 'DNS', 'DNF', 'DNQ', 'DQ')
AND RESULT IS NOT NULL"""

all_sql="""
SELECT * FROM `saa-analytics.results.PRODUCTION`
"""

## Read all performance benchmarks csv from GCS bucket and process##
# Benchmark column names must be BENCHMARK_COMPETITION, EVENT, GENDER, RESULT_BENCHMARK, STANDARDISED_BENCHMARK, 2%, 3.50%, 5%, 10%


@st.cache_data(ttl=400)
def fetch_benchmarks():
    conn = st.connection('gcs', type=FilesConnection, ttl=600)
    benchmarks = conn.read("competition_benchmarks/All_Benchmarks_Processed.csv", input_format="csv")
    return benchmarks
benchmarks = fetch_benchmarks()  # fetch benchmarks

## Download all athlete data from BQ

@st.cache_data(ttl=400)
def fetch_data():  # fetch athlete results
    data = client.query_and_wait(athletes_sql).to_dataframe()

    data.dropna(how= "all", axis=1, inplace=True)

    # DATE column to contain timezone - tz aware mode

    data['DATE'] = pd.to_datetime(data['DATE'], format='mixed', dayfirst=False, utc=True)

    # datetime to contain UTC (timezone)

    data['NOW'] = datetime.datetime.now()
    timezone = pytz.timezone('UTC')
    data['NOW'] = datetime.datetime.now().replace(tzinfo=timezone)

    data['delta_time'] = data['NOW'] - data['DATE']
    data['delta_time_conv'] = pd.to_numeric(data['delta_time'].dt.days, downcast='integer')
    data['event_month'] = data['DATE'].dt.month

    
    data['MAPPED_EVENT']=''

    map_international_events(data) # call function to map relevant events

 #   process_results(data) # convert results into seconds format

    return data

@st.cache_data(ttl=600)
def fetch_all_data():  # fetch athlete results
    
    all_data = client.query_and_wait(all_sql).to_dataframe()

    return all_data

## Get all the data ##

data = fetch_data() # fetch the database of results for selected period
all_data = fetch_all_data() # fetch the entire database

#data['DATE'] = pd.to_datetime(data['DATE'], format='mixed', dayfirst=False, utc=True)

# datetime to contain UTC (timezone)

#data['NOW'] = datetime.datetime.now()
#timezone = pytz.timezone('UTC')
#data['NOW'] = datetime.datetime.now().replace(tzinfo=timezone)

#data['delta_time'] = data['NOW'] - data['DATE']
#data['delta_time_conv'] = pd.to_numeric(data['delta_time'].dt.days, downcast='integer')
#data['event_month'] = data['DATE'].dt.month

#data['DATE'] = data['DATE'].dt.tz_localize(None)  # switch off timezone for compatibility with np.datetime64

## Convert DATE to datetime with timezone ##


# Make sure date conversion is is valid for all rows

#assert not competitors['delta_time'].isna().any()


benchmark_option = st.selectbox(
    "  ",
    ("Search Database Records by Name or Competition", "2023 SEAG Bronze - SEAG Selection", 
     "2023 SEAG Bronze - OCTC Selection", "26th Asian Athletics", "2025 Taiwan Open")
)

st.write(' ')

if benchmark_option == 'Search Database Records by Name or Competition':

    search_option = st.selectbox(
    "Select Your Search Option:",
    options=['Athlete Name', 'Competition Name'],
    )
    if search_option=='Athlete Name':
    
        text_search = st.text_input("Enter Search Keyword for Athlete Name", value="")
        text = text_search.casefold()

        all_data['NAME'] = all_data['NAME'].str.casefold()  # convert everything to lower case (NEW)

    
        names['VARIATION'] = names['VARIATION'].str.casefold()
        names['NAME'] = names['NAME'].str.casefold()

        for row in names.itertuples():  # itertuples is faster
        
            all_data['NAME'] = all_data['NAME'].replace(regex=rf"{row.VARIATION}", value=f"{row.NAME}")   

        
        all_data['NAME'] = all_data['NAME'].str.title()  # capitalize first letter (NEW)

        all_data['NAME_case'] = all_data["NAME"].str.casefold()

        combinations = all_data[all_data['NAME'].str.casefold().str.contains(text)]['NAME'].unique().tolist()
 
        #  name_selected = st.multiselect('Select From List of Matches :', all_data.loc[all_data['NAME_case'].str.contains(text)]['NAME'].unique())
        name_selected = st.multiselect('Select From List of Matches :', combinations)

        st.write(name_selected[0])

      #  st.write(name_selected[0])
        m1 = all_data["NAME"].str.contains(name_selected[0])

        
        #all_data.loc[all_data['NAME_case'].str.contains(text)]['NAME_case'].unique()
        df_search = all_data[m1].sort_values(by='DATE', ascending=False)

        if text_search:
            st.write(df_search)

    #    all_data.drop(['NAME_case'], axis=1, inplace=True)
#
    

    elif search_option=='Competition Name':

        text_search = st.text_input("Enter Search Keyword for Competition Name", value="")
        text = text_search.casefold()

        all_data['COMPETITION_case'] = all_data["COMPETITION"].str.casefold()
        m2 = all_data["COMPETITION_case"].str.contains(text)
        df_search = all_data[m2]

        if text_search:
            st.write(df_search)

        all_data.drop(['COMPETITION_case'], axis=1, inplace=True)


    # Show the results, if you have a text_search
    
    
#    final_dfs, code = spreadsheet(all_data)
  #  st.write(final_dfs)

    # Allow text search on athlete name and/or competition
    
    
else:

    # Choose start and end dates
    
    start_date = st.date_input("Input Start Period (dd/mm/yyyy)", format = 'DD/MM/YYYY')
    end_date = st.date_input("Input End Period (dd/mm/yyy)", format = 'DD/MM/YYYY') 

    start = np.datetime64(start_date)
    end = np.datetime64(end_date)

    data['DATE'] = pd.to_datetime(data['DATE'], format='mixed', dayfirst=False, utc=True)
    data['DATE'] = data['DATE'].dt.tz_localize(None)  # switch off timezone for compatibility with np.datetime64
    

    mask = ((data['DATE'] >= start) & (data['DATE'] <= end))
    athletes_selected = data.loc[mask]
    athletes_selected.reset_index(drop=True, inplace=True)

    # Fast lookup for benchmarks
    bench_map = {
        "2023 SEAG Bronze - SEAG Selection": '2023 SEAG Bronze',
        "2023 SEAG Bronze - OCTC Selection": '2023 SEAG Bronze',
        "26th Asian Athletics": '26th Asian Athletics',
        "2025 Taiwan Open": '2025 Taiwan Open'
    }
    bench_name = bench_map.get(benchmark_option, None)
    benchmark = benchmarks[benchmarks['BENCHMARK_COMPETITION'] == bench_name] if bench_name else pd.DataFrame()

## Map relevant events to a standard description ##

#athletes_selected['MAPPED_EVENT']=''

#map_international_events(athletes_selected) # call function

## Override selection of athletes for 2025 World Athletics Champs only ##

#if benchmark_option == '2025 World Athletics Champs':

   # data=data.reset_index(drop=True)
    
    
#    start_1 = datetime.datetime(2023, 11, 5)
#    end_1 = datetime.datetime(2025, 5, 4)
    
#    start_2 = datetime.datetime(2024, 2, 25)
#    end_2 = datetime.datetime(2025, 8, 24)
    
#    start_3 = datetime.datetime(2024, 8, 1)
#    end_3 = datetime.datetime(2025, 8, 24)
    
    
#    start_date1 = np.datetime64(start_1)
#    end_date1 = np.datetime64(end_1)
#    start_date2 = np.datetime64(start_2)
#    end_date2 = np.datetime64(end_2)
#    start_date3 = np.datetime64(start_3)
#    end_date3 = np.datetime64(end_3)

 #   data['MAPPED_EVENT']=''

 #   map_international_events(data) # call function

  #  mask1 = ((data['MAPPED_EVENT']==(('Marathon') or ('35km Racewalk'))) & (data['event_date_dt'] >= start_date1) & (data['event_date_dt'] <= end_date1))
 #   mask2 = ((data['MAPPED_EVENT']==(('10,000m') or ('20km Racewalk') or ('Combined'))) & (data['event_date_dt'] >= start_date2) & (data['event_date_dt'] <= end_date2))
 #   mask3 = ((data['MAPPED_EVENT']!=(('Marathon') or ('35km Racewalk') or ('10,000m')|('20km Racewalk')|('Combined'))) & (data['event_date_dt'] >= start_date3) & (data['event_date_dt'] <= end_date3))
    
  #  combined_mask = (mask1|mask2|mask3)
    
  #  athletes_selected = data.loc[combined_mask]

## Map benchmarks ##

#st.dataframe(athletes_selected)

if benchmark_option != 'Search Database Records by Name or Competition':

    df = pd.merge(
        left=athletes_selected, 
        right=benchmark,
        how='left',
        left_on=['MAPPED_EVENT', 'GENDER'],
        right_on=['EVENT', 'GENDER'],
    )                   
    
    
    clean_columns(df) # clean benchmarks of hidden characters, spaces etc. to ensure proper merging

    
    #st.write(df.columns)


    df['RESULT'] = df['RESULT'].replace(regex=r'–', value=np.nan)

    process_results(df) # call function to convert results to standard float64 format


## Create scalar to measure relative performance - distance events are reversed from timed events ##

    df['PERF_SCALAR']=df['Delta5']/df['STANDARDISED_BENCHMARK']*100

# Name corrections
# Read name variations from GCS name lists bucket (Still in beta)

# Read csv of name variations from GCS bucket

 #   conn = st.connection('gcs', type=FilesConnection, ttl=600)
  #  names = conn.read("name_variations/name_variations.csv", input_format="csv")

  #  names = clean_columns(names)  # clean name list of special characters, white spaces etc.

    

# Iterate over dataframe and replace names using casefold then convert to capitalize first letter

    df['NAME'] = df['NAME'].str.casefold()  # convert everything to lower case (NEW)

    
    names['VARIATION'] = names['VARIATION'].str.casefold()
    names['NAME'] = names['NAME'].str.casefold()

    for row in names.itertuples():  # itertuples is faster
        
        df['NAME'] = df['NAME'].replace(regex=rf"{row.VARIATION}", value=f"{row.NAME}")   

    df['NAME'] = df['NAME'].str.title()  # capitalize first letter (NEW)


    
# Remove foreigners

    df = df.loc[~df['NAME'].str.casefold().isin(exclusions)]  # ~ means NOT IN. DROP spex carded athletes

    
# Choose the best result for each event participated by every athlete


#st.write(foreigners)


# Process list of foreign names and their variations

    # Remove foreign/national teams and nationalities efficiently
    
    excluded_teams = {
        'Malaysia','THAILAND','China','South Korea','Laos','Myanmar','Philippines',
        'Piboonbumpen Thailand','Chinese Taipei','Gurkha Contingent','Australia','Hong Kong',
        'PERAK','Sri Lanka','Indonesia','Waseda','Vietnam','INDIA','Hong Kong, China','AIC JAPAN'
    }
    excluded_nationalities = {'GBR','IND','MAS','INA','JPN','SRI','THA'}
    df_local_teams = df[~df['TEAM'].isin(excluded_teams) & ~df['NATIONALITY'].isin(excluded_nationalities)]

 #   st.write('Only Locals')
 #   st.write(len(df_local_teams))

    

    # Find out top performance for each athlete and event
    
    top_performers_clean = df_local_teams.sort_values(['MAPPED_EVENT', 'NAME','PERF_SCALAR'],ascending=False).groupby(['MAPPED_EVENT', 'NAME']).head(1)
    
    top_performers_clean.reset_index(inplace=True, drop=True)


    
#    st.write('Top Athlete Result')
#    st.write(len(top_performers_clean))


    # Create performance tier column
    
    top_performers_clean['TIER'] = np.where((top_performers_clean['Delta_Benchmark']>=0), 'Tier 1',    
                                    np.where(((top_performers_clean['Delta_Benchmark']<0) & (top_performers_clean['Delta2']>=0)), 'Tier 2',
                                    np.where(((top_performers_clean['Delta2']<0) & (top_performers_clean['Delta3.5']>=0)), 'Tier 3', 
                                    np.where(((top_performers_clean['Delta3.5']<0) & (top_performers_clean['Delta5']>=0)), 'Tier 4', 
                                    np.where(((top_performers_clean['Delta5']<0) & (top_performers_clean['Delta10']>=0)), 'Tier 5', ' ')))))
    
    
    # Drop rows without a corresponding benchmark

    #st.write(top_performers_clean.columns)
    
    df_no_na = top_performers_clean[top_performers_clean['STANDARDISED_BENCHMARK'].notna()]
    df_no_na = df_no_na[['NAME', 'COMPETITION_RANK', 'TEAM', 'RESULT', 'WIND', 'EVENT_x', 'DIVISION', 'STAGE', 'AGE', 'GENDER', 'UNIQUE_ID', 'NATIONALITY', 'DICT_RESULTS', 'DATE', 'YEAR', 'COMPETITION', 'DOB', 'CATEGORY_EVENT',
                        'REGION', 'SOURCE', 'REMARKS', 'SUB_EVENT', 'DISTANCE', 'MAPPED_EVENT', 'BENCHMARK_COMPETITION', 'RESULT_BENCHMARK', 'STANDARDISED_BENCHMARK', '2%', '3.50%', '5%',
                        '10%', 'RESULT_CONV', 'Delta2', 'Delta3.5', 'Delta5', 'Delta10', 'Delta_Benchmark', 'PERF_SCALAR', 'TIER']]
    
    df_no_na = df_no_na.reindex(columns= ['NAME', 'COMPETITION_RANK', 'TEAM', 'RESULT', 'RESULT_CONV', 'WIND', 'EVENT_x', 'MAPPED_EVENT', 'CATEGORY_EVENT', 'SUB_EVENT', 'DISTANCE', 'DIVISION', 'STAGE', 'AGE',  'DOB', 'GENDER', 'UNIQUE_ID', 'NATIONALITY', 'DICT_RESULTS', 'YEAR', 'DATE', 'COMPETITION',
                        'REGION', 'SOURCE', 'REMARKS', 'BENCHMARK_COMPETITION', 'RESULT_BENCHMARK', 'STANDARDISED_BENCHMARK', '2%', '3.50%', '5%',
                        '10%', 'Delta2', 'Delta3.5', 'Delta5', 'Delta10', 'Delta_Benchmark', 'PERF_SCALAR', 'TIER'])

  #  st.write('Tiered')
  #  st.write(len(df_no_na))

   

    if benchmark_option == '2023 SEAG Bronze - OCTC':   # Additional logic for OCTC report

        # Rank everyone for octc selection

        all_ranking_octc = final_df.sort_values(['MAPPED_EVENT','GENDER','PERF_SCALAR'], ascending=[False, False, False])
        all_ranking_octc['Rank'] = all_ranking_octc.groupby(['GENDER', 'MAPPED_EVENT', 'TIER']).cumcount() + 1

        all_ranking_octc['TIER_ADJ'] = np.where(
                                ((all_ranking_octc['TIER']=='Tier 1') & (all_ranking_octc['Rank']==3)), 'Tier 2',    
                                np.where(
                                ((all_ranking_octc['TIER']=='Tier 1') & (all_ranking_octc['Rank']>=4)), 'Tier 2',
                                np.where(
                                ((all_ranking_octc['TIER']=='Tier 2') & (all_ranking_octc['Rank']==3)), 'Tier 3', 
                                np.where(
                                ((all_ranking_octc['TIER']=='Tier 2') & (all_ranking_octc['Rank']>=4)), 'Tier 3', 
                                np.where(                             
                                ((all_ranking_octc['TIER']=='Tier 3') & (all_ranking_octc['Rank']==3)), 'Tier 4', 
                                np.where(                             
                                ((all_ranking_octc['TIER']=='Tier 3') & (all_ranking_octc['Rank']>=4)), 'Tier 4', all_ranking_octc['TIER']) 
                
                                )))))

        

        rerank_octc = all_ranking_octc.sort_values(['MAPPED_EVENT','GENDER','TIER_ADJ', 'PERF_SCALAR'], ascending=[False, False, False, False])
        rerank_octc['Rank_ADJ'] = rerank_octc.groupby(['MAPPED_EVENT', 'GENDER', 'TIER_ADJ']).cumcount() + 1

        rerank_filtered_octc = rerank_octc[(rerank_octc['TIER_ADJ']!=' ') & (rerank_octc['TIER_ADJ']!='Tier 4')]

        rerank_filtered_octc = rerank_filtered_octc.drop(['TIER', 'Rank'], axis=1)

        rerank_filtered_octc.rename(columns={'TIER_ADJ': 'TIER', 'Rank_ADJ': 'TIER_RANKING'}, inplace=True)

        
        final_df = rerank_filtered_octc



        
    
# Show resulting OCTC dataframe
    st.write(" ")
    st.write(" ")
    st.write("LIST OF ATHLETES MEETING BENCHMARKS:")
    st.write(" ")
    st.write(" ")



    final_df = df_no_na[df_no_na['TIER']!=' ']  # Choose only those record with Tier value
    

    final_df=final_df.reset_index(drop=True)

    # Use a text_input to get the keywords to filter the dataframe


    
    st.write(final_df)



