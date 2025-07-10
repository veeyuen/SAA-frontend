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

foreigners['V1'] = foreigners['LAST_NAME']+' '+foreigners['FIRST_NAME']
foreigners['V2'] = foreigners['FIRST_NAME']+' '+foreigners['LAST_NAME']
foreigners['V3'] = foreigners['LAST_NAME']+', '+foreigners['FIRST_NAME']
foreigners['V4'] = foreigners['FIRST_NAME']+' '+foreigners['LAST_NAME']

for1 = foreigners['V1'].dropna().tolist()
for2 = foreigners['V2'].dropna().tolist()
for3 = foreigners['V3'].dropna().tolist()
for4 = foreigners['V4'].dropna().tolist()

foreign_list = for1+for2+for3+for4 

foreign_list_casefold=[s.casefold() for s in foreign_list]

exclusions = foreign_list_casefold
    
### DEFINE SQL QUERIES ###

athletes_sql="""
SELECT NAME, RESULT, TEAM, AGE, RANK AS COMPETITION_RANK, STAGE, DICT_RESULTS, SOURCE, REMARKS, SUB_EVENT,  DIVISION, EVENT, DATE, DISTANCE, EVENT_CLASS, UNIQUE_ID, DOB, NATIONALITY, WIND, CATEGORY_EVENT, GENDER, COMPETITION, YEAR, REGION
FROM `saa-analytics.results.PRODUCTION` 
WHERE RESULT NOT IN ('NM', '-', 'DNS', 'DNF', 'DNQ', 'DQ')
AND RESULT IS NOT NULL"""

all_sql="""
SELECT * FROM `saa-analytics.results.PRODUCTION`
"""

athletes_map_sql="""

SELECT NAME, RESULT, TEAM, AGE, RANK AS COMPETITION_RANK, DIVISION, EVENT, DISTANCE, EVENT_CLASS, UNIQUE_ID, DOB, NATIONALITY, WIND, CATEGORY_EVENT, GENDER, COMPETITION, YEAR, REGION
FROM `saa-analytics.results.PRODUCTION` 
WHERE RESULT NOT IN ('NM', '-', 'DNS', 'DNF', 'DNQ', 'DQ')
AND RESULT IS NOT NULL

-- Standard running events
UPDATE athletes SET MAPPED_EVENT = '100m'
WHERE EVENT REGEXP '(Dash|Run).*100|100 Meter Run|^100m$';

UPDATE athletes SET MAPPED_EVENT = '200m'
WHERE EVENT REGEXP '(Dash|Run).*200|200 Meter Run|^200m$|200\\sMeter';

UPDATE athletes SET MAPPED_EVENT = '400m'
WHERE EVENT REGEXP '(Dash|Run).*400|^400m$|^400\\sMeter$|400m Relay';

UPDATE athletes SET MAPPED_EVENT = '800m'
WHERE EVENT REGEXP '(Run).*800|800 Meter Run|^800m$';

UPDATE athletes SET MAPPED_EVENT = '1000m'
WHERE EVENT REGEXP '(Run).*1000';

UPDATE athletes SET MAPPED_EVENT = '1500m'
WHERE EVENT REGEXP '(Run).*1500|^1500m$';

UPDATE athletes SET MAPPED_EVENT = '3000m'
WHERE EVENT REGEXP '(Run).*3000';

UPDATE athletes SET MAPPED_EVENT = '5000m'
WHERE EVENT REGEXP '(Run).*5000|^5000m$';

UPDATE athletes SET MAPPED_EVENT = '10,000m'
WHERE EVENT REGEXP '(Run).*10000|^10000m$|^10,000m$';

UPDATE athletes SET MAPPED_EVENT = '1 Mile'
WHERE EVENT REGEXP '(Run).*Mile';

-- Throws
UPDATE athletes SET MAPPED_EVENT = 'Javelin Throw'
WHERE EVENT LIKE '%Javelin%';

UPDATE athletes SET MAPPED_EVENT = 'Shot Put'
WHERE EVENT REGEXP 'Shot Put|Shot put';

UPDATE athletes SET MAPPED_EVENT = 'Hammer Throw'
WHERE EVENT REGEXP 'Hammer Throw|Hammer throw';

UPDATE athletes SET MAPPED_EVENT = 'Discus Throw'
WHERE EVENT REGEXP 'Discus Throw|Discus|Discus throw';

-- Jumps
UPDATE athletes SET MAPPED_EVENT = 'High Jump'
WHERE EVENT REGEXP 'High Jump|High jump';

UPDATE athletes SET MAPPED_EVENT = 'Long Jump'
WHERE EVENT REGEXP 'Long Jump|Long jump';

UPDATE athletes SET MAPPED_EVENT = 'Triple Jump'
WHERE EVENT REGEXP 'Triple Jump|Triple jump';

UPDATE athletes SET MAPPED_EVENT = 'Pole Vault'
WHERE EVENT REGEXP 'Pole Vault|Pole vault';

-- Steeplechase
UPDATE athletes SET MAPPED_EVENT = '3000m Steeplechase'
WHERE EVENT REGEXP '3000m Steeplechase|3000m S/C|Steeplechase|S/C';

-- Marathons
UPDATE athletes SET MAPPED_EVENT = 'Marathon'
WHERE EVENT = 'Marathon';

UPDATE athletes SET MAPPED_EVENT = 'Half Marathon'
WHERE EVENT IN ('Half Marathon', 'Half marathon');

-- Race walk
UPDATE athletes SET MAPPED_EVENT = '10000m Racewalk'
WHERE EVENT LIKE '%Race Walk%' AND DISTANCE LIKE '%10000%';

-- Relays
UPDATE athletes SET MAPPED_EVENT = '4 x 80m'
WHERE EVENT = '4x80m Relay';

UPDATE athletes SET MAPPED_EVENT = '4 x 100m'
WHERE EVENT IN ('4x100m Relay', '4 X 100m Relay', '4x100 Meter Relay', '4 x 100m') OR (EVENT LIKE '%Relay%' AND DISTANCE LIKE '%400%');

UPDATE athletes SET MAPPED_EVENT = '4 x 400m'
WHERE EVENT IN ('4x400m Relay', '4 X 400m Relay', '4 x 400m') OR (EVENT LIKE '%Relay%' AND DISTANCE LIKE '%1600%');

-- Combined events
UPDATE athletes SET MAPPED_EVENT = 'Heptathlon'
WHERE EVENT IN ('Heptathlon');

UPDATE athletes SET MAPPED_EVENT = 'Decathlon'
WHERE EVENT IN ('Decathlon');

-- Hurdles (examples, further logic can be expanded based on your specific rules)
-- 100m Hurdles (Women)
UPDATE athletes SET MAPPED_EVENT = '100m Hurdles'
WHERE (EVENT LIKE '%100m Hurdles%' OR EVENT LIKE '%100m hurdles%' OR (EVENT = 'Hurdles' AND DISTANCE = '100'))
  AND GENDER = 'Female';

-- 110m Hurdles (Men)
UPDATE athletes SET MAPPED_EVENT = '110m Hurdles'
WHERE (EVENT LIKE '%110m Hurdles%' OR EVENT LIKE '%110m hurdles%' OR (EVENT = 'Hurdles' AND DISTANCE = '110'))
  AND GENDER = 'Male';

-- 400m Hurdles
UPDATE athletes SET MAPPED_EVENT = '400m Hurdles'
WHERE (EVENT LIKE '%400m Hurdles%' OR EVENT LIKE '%400m hurdles%' OR (EVENT = 'Hurdles' AND DISTANCE = '400'));

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
    data = client.query_and_wait(athletes_map_sql).to_dataframe()

    data.dropna(how= "all", axis=1, inplace=True)

    # DATE column to contain timezone - tz aware mode

#    data['DATE'] = pd.to_datetime(data['DATE'], format='mixed', dayfirst=False, utc=True)

    # datetime to contain UTC (timezone)

#    data['NOW'] = datetime.datetime.now()
#    timezone = pytz.timezone('UTC')
#    data['NOW'] = datetime.datetime.now().replace(tzinfo=timezone)

#    data['delta_time'] = data['NOW'] - data['DATE']
#    data['delta_time_conv'] = pd.to_numeric(data['delta_time'].dt.days, downcast='integer')
#    data['event_month'] = data['DATE'].dt.month

#    data['DATE'] = data['DATE'].dt.tz_localize(None)  # switch off timezone for compatibility with np.datetime64
    
#    data['MAPPED_EVENT']=''

#    map_international_events(data) # call function

    return data

@st.cache_data(ttl=400)
def fetch_all_data():  # fetch athlete results
    
    all_data = client.query_and_wait(all_sql).to_dataframe()

    return all_data

## Get all the data ##

data = fetch_data() # fetch the database of results for selected period
all_data = fetch_all_data() # fetch the entire database

data['DATE'] = pd.to_datetime(data['DATE'], format='mixed', dayfirst=False, utc=True)

# datetime to contain UTC (timezone)

data['NOW'] = datetime.datetime.now()
timezone = pytz.timezone('UTC')
data['NOW'] = datetime.datetime.now().replace(tzinfo=timezone)

data['delta_time'] = data['NOW'] - data['DATE']
data['delta_time_conv'] = pd.to_numeric(data['delta_time'].dt.days, downcast='integer')
data['event_month'] = data['DATE'].dt.month

data['DATE'] = data['DATE'].dt.tz_localize(None)  # switch off timezone for compatibility with np.datetime64

## Convert DATE to datetime with timezone ##


# Make sure date conversion is is valid for all rows

#assert not competitors['delta_time'].isna().any()

start_date = st.date_input("Input Start Period (dd/mm/yyyy)", format = 'DD/MM/YYYY')
end_date = st.date_input("Input End Period (dd/mm/yyy)", format = 'DD/MM/YYYY') 

start = np.datetime64(start_date)
end = np.datetime64(end_date)


mask = ((data['DATE'] >= start) & (data['DATE'] <= end))
athletes_selected = data.loc[mask]


## Allow public access via mito

#final_dfs, code = spreadsheet(athletes_selected)

benchmark_option = st.selectbox(
    "Please Select Performance Benchmark (Select 'None' to Access All Records in Database)",
    ("None - Direct Access to All Database Records", "2023 SEAG Bronze - SEAG Selection", "2023 SEAG Bronze - OCTC Selection", "26th Asian Athletics", "2025 Taiwan Open"),
)

if benchmark_option == 'None - Direct Access to All Database Records':

    final_dfs, code = spreadsheet(all_data)

    benchmark = pd.DataFrame()

elif benchmark_option == '2023 SEAG Bronze - SEAG Selection' or benchmark_option == '2023 SEAG Bronze - OCTC Selection':

    benchmark = benchmarks[benchmarks['BENCHMARK_COMPETITION']== '2023 SEAG Bronze']


elif benchmark_option == '26th Asian Athletics':

    benchmark = benchmarks[benchmarks['BENCHMARK_COMPETITION']== '26th Asian Athletics']


elif benchmark_option == '2025 Taiwan Open':

    benchmark = benchmarks[benchmarks['BENCHMARK_COMPETITION']== '2025 Taiwan Open']

elif benchmark_option == '2025 World Athletics Champs':

    benchmark = benchmarks[benchmarks['BENCHMARK_COMPETITION']== '2025 World Athletics Champs']



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

if benchmark_option != 'None - Direct Access to All Database Records':

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

    names['VARIATION'] = names['VARIATION'].str.casefold() # convert to lower case
    names['NAME'] = names['NAME'].str.casefold()


# Iterate over dataframe and replace names
    
    df['NAME'] = df['NAME'].str.casefold()  # everything lower case

  #  for index, row in names.iterrows():
        
  #      df['NAME'] = df['NAME'].replace(regex=rf"{row['VARIATION']}", value=f"{row['NAME']}")

    for row in names.itertuples():  # itertuples is faster
        
        df['NAME'] = df['NAME'].replace(regex=rf"{row.VARIATION}", value=f"{row.NAME}")   

    
    df['NAME'] = df['NAME'].str.title()  # capitalize first letter

# Remove foreigners

    df = df.loc[~df['NAME'].str.casefold().isin(exclusions)]  # ~ means NOT IN. DROP spex carded athletes

# Choose the best result for each event participated by every athlete

#top_performers = no_foreigners_list.sort_values(['MAPPED_EVENT', 'NAME','PERF_SCALAR'],ascending=False).groupby(['MAPPED_EVENT', 'NAME']).head(1)

#st.write(foreigners)


# Process list of foreign names and their variations

    df_local_teams = df[(df['TEAM']!='Malaysia')&(df['TEAM']!='THAILAND')&(df['TEAM']!='China') 
                           &(df['TEAM']!='South Korea')&(df['TEAM']!='Laos')&(df['TEAM']!='Thailand')&(df['TEAM']!='Myanmar') 
                           &(df['TEAM']!='Philippines')&(df['TEAM']!='Piboonbumpen Thailand') 
                           &(df['TEAM']!='Chinese Taipei')&(df['TEAM']!='Gurkha Contingent') 
                           &(df['TEAM']!='Australia')&(df['TEAM']!='Piboonbumpen Thailand') 
                           &(df['TEAM']!='Hong Kong')&(df['TEAM']!='PERAK')&(df['TEAM']!='Sri Lanka') 
                           &(df['TEAM']!='Indonesia')&(df['TEAM']!='THAILAND')&(df['TEAM']!='MALAYSIA') 
                           &(df['TEAM']!='PHILIPPINES') & (df['TEAM']!='SOUTH KOREA')&(df['TEAM']!='Waseda') 
                           &(df['TEAM']!='LAOS')&(df['TEAM']!='CHINESE TAIPEI')&(df['TEAM']!='Vietnam')
                           &(df['TEAM']!='INDIA')&(df['TEAM']!='Hong Kong, China')&(df['TEAM']!='AIC JAPAN')
                           &(df['NATIONALITY']!='GBR')&(df['NATIONALITY']!='IND')&(df['NATIONALITY']!='MAS')&(df['NATIONALITY']!='INA')&(df['NATIONALITY']!='JPN')
                           &(df['NATIONALITY']!='SRI')&(df['NATIONALITY']!='THA')] 

    top_performers_clean = df_local_teams.sort_values(['MAPPED_EVENT', 'NAME','PERF_SCALAR'],ascending=False).groupby(['MAPPED_EVENT', 'NAME']).head(1)
    
    top_performers_clean.reset_index(inplace=True, drop=True)
    
    
    
    # Create performance tier column
    
    top_performers_clean['TIER'] = np.where((top_performers_clean['Delta_Benchmark']>=0), 'Tier 1',    
                                    np.where(((top_performers_clean['Delta_Benchmark']<0) & (top_performers_clean['Delta2']>=0)), 'Tier 2',
                                    np.where(((top_performers_clean['Delta2']<0) & (top_performers_clean['Delta3.5']>=0)), 'Tier 3', 
                                    np.where(((top_performers_clean['Delta3.5']<0) & (top_performers_clean['Delta5']>=0)), 'Tier 4', 
                                    np.where(((top_performers_clean['Delta5']<0) & (top_performers_clean['Delta10']>=0)), 'Tier 5', ' ')))))
    
    
    # Drop rows without a corresponding benchmark

    #st.write(top_performers_clean.columns)
    
    final_df = top_performers_clean[top_performers_clean['STANDARDISED_BENCHMARK'].notna()]
    final_df = final_df[['NAME', 'COMPETITION_RANK', 'TEAM', 'RESULT', 'WIND', 'EVENT_x', 'DIVISION', 'STAGE', 'AGE', 'GENDER', 'UNIQUE_ID', 'NATIONALITY', 'DICT_RESULTS', 'DATE', 'YEAR', 'COMPETITION', 'DOB', 'CATEGORY_EVENT',
                        'REGION', 'SOURCE', 'REMARKS', 'SUB_EVENT', 'DISTANCE', 'MAPPED_EVENT', 'BENCHMARK_COMPETITION', 'RESULT_BENCHMARK', 'STANDARDISED_BENCHMARK', '2%', '3.50%', '5%',
                        '10%', 'RESULT_CONV', 'Delta2', 'Delta3.5', 'Delta5', 'Delta10', 'Delta_Benchmark', 'PERF_SCALAR', 'TIER']]
    
    final_df = final_df.reindex(columns= ['NAME', 'COMPETITION_RANK', 'TEAM', 'RESULT', 'RESULT_CONV', 'WIND', 'EVENT_x', 'MAPPED_EVENT', 'CATEGORY_EVENT', 'SUB_EVENT', 'DISTANCE', 'DIVISION', 'STAGE', 'AGE',  'DOB', 'GENDER', 'UNIQUE_ID', 'NATIONALITY', 'DICT_RESULTS', 'YEAR', 'DATE', 'COMPETITION',
                        'REGION', 'SOURCE', 'REMARKS', 'BENCHMARK_COMPETITION', 'RESULT_BENCHMARK', 'STANDARDISED_BENCHMARK', '2%', '3.50%', '5%',
                        '10%', 'Delta2', 'Delta3.5', 'Delta5', 'Delta10', 'Delta_Benchmark', 'PERF_SCALAR', 'TIER'])


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

        #final_df = rerank_filtered_octc[['NAME', 'RANK', 'TEAM', 'RESULT', 'WIND', 'EVENT_x', 'DIVISION', 'STAGE', 'AGE', 'GENDER', 'UNIQUE_ID', 'NATIONALITY', 'DICT_RESULT', 'YEAR', 'COMPETITION', 'DOB', 'CATEGORY_EVENT',
        #                                'REGION', 'SOURCE', 'REMARKS', 'SUB_EVENT', 'DISTANCE', 'event_date_dt', 'MAPPED_EVENT', 'BENCHMARK_COMPETITION', 'RESULT_BENCHMARK', 'STANDARDISED_BENCHMARK', '2%', '3.50%', '5%',
        #                                '10%', 'RESULT_CONV', 'Delta2', 'Delta3.5', 'Delta5', 'Delta10', 'Delta_Benchmark', 'PERF_SCALAR', 'TIER']]

        final_df = rerank_filtered_octc



        
    
# Show resulting OCTC dataframe
    st.write(" ")
    st.write(" ")
    st.write("LIST OF ATHLETES MEETING BENCHMARKS:")
    st.write(" ")
    st.write(" ")



    final_df = final_df[final_df['TIER']!=' ']  # Choose only those record with Tier value
   # final_df = final_df.loc[:, ['NAME', 'RANK', 'TEAM', 'RESULT', 'QUALIFICATION', 'WIND', 'DIVISION', 'STAGE', 'POINTS', 'AGE', 'GENDER', 'UNIQUE_ID', 'NATIONALITY',
   ## 'DICT_RESULTS', 'COMPETITION', 'REGION', 'DOB', 'CATEGORY_EVENT', 'SOURCE', 'REMARKS', 'SUB_EVENT', 'SESSION', 'EVENT_CLASS', 'event_date_dt',
   # 'MAPPED_EVENT', 'BENCHMARK_COMPETITION', 'STANDARDISED_BENCHMARK', '2%', '3.50%', '5%', 'RESULT_CONV', 'Delta2', 'Delta3.5', 'Delta5', 'Delta_Benchmark', 'PERF_SCALAR', 'TIER']]

    final_df=final_df.reset_index(drop=True)

    
    st.write(final_df)
