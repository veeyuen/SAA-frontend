# streamlit_BQ.py
# OCTC Selection for SAA Athletes

import streamlit as st
import pandas as pd
import numpy as np
import datetime
import db_dtypes
import analytics
import re
import gcsfs
import pytz
import math
from streamlit_gsheets import GSheetsConnection
from st_files_connection import FilesConnection
from functions import convert_time, process_results, map_international_events, clean_columns, simple_map_events, normalize_text
from functions import normalize_time_format, convert_time_refactored, convert_time_format, convert_time_refactored_2
from google.cloud import storage
from mitosheet.streamlit.v1 import spreadsheet

st.set_page_config(layout="wide")

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
@st.cache_data(ttl=6000)
def fetch_foreigners():
    conn = st.connection('gcs', type=FilesConnection, ttl=600)
    foreigners = conn.read("name_lists/List of Foreigners.csv", encoding="utf-8", input_format="csv")
    return foreigners
foreigners = fetch_foreigners()  # get list of foreigners

@st.cache_data(ttl=20000)
def name_variations():
    conn = st.connection('gcs', type=FilesConnection, ttl=0)
    names = conn.read("name_variations/name_variations.csv", input_format="csv")
    names = clean_columns(names)  # clean name list of special characters, white spaces etc.
    return names
#names = name_variations()
#st.write(names)

# Load google sheet with name variations

@st.cache_data(ttl=200)
def gspread_names():

    conn = st.connection("gsheets", type=GSheetsConnection)
    # Select a specific worksheet
    data = conn.read()
    # Get all values from the worksheet as a list of lists
    names = pd.DataFrame(data)
    names = clean_columns(names)  # clean name list of special characters, white spaces etc.

    return names

names = gspread_names()
#st.write(names)

    



# Create list of foreigners 


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
FROM `saa-analytics.results.PRODUCTION_CORRECT` 
WHERE RESULT NOT IN ('NM', '-', 'DNS', 'DNF', 'DNQ', 'DQ')
AND RESULT IS NOT NULL"""

all_sql="""
SELECT NAME, EVENT, DATE, COMPETITION, RESULT, WIND, HOST_CITY, TEAM, AGE, GENDER, SUB_EVENT, DIVISION, DISTANCE, EVENT_CLASS, DOB, NATIONALITY
FROM `saa-analytics.results.PRODUCTION_CORRECT`
"""

## Read all performance benchmarks csv from GCS bucket and process##
# Benchmark column names must be BENCHMARK_COMPETITION, EVENT, GENDER, RESULT_BENCHMARK, STANDARDISED_BENCHMARK, 2%, 3.50%, 5%, 10%


@st.cache_data(ttl=6000)
def fetch_benchmarks():
    conn = st.connection('gcs', type=FilesConnection, ttl=600)
    benchmarks = conn.read("competition_benchmarks/All_Benchmarks_Processed.csv", input_format="csv")
    return benchmarks
benchmarks = fetch_benchmarks()  # fetch benchmarks

## Download all athlete data from BQ

@st.cache_data(ttl=6000)
def fetch_data():  # for reports
    data = client.query_and_wait(athletes_sql).to_dataframe()

    data.dropna(how= "all", axis=1, inplace=True)

    # DATE column to contain timezone - tz aware mode

    data['DATE'] = pd.to_datetime(data['DATE'], format='mixed', dayfirst=False, utc=True)

    # datetime to contain UTC (timezone)

    timezone = pytz.timezone('UTC')
    data['NOW'] = datetime.datetime.now().replace(tzinfo=timezone)

    data['delta_time'] = data['NOW'] - data['DATE']
    data['delta_time_conv'] = pd.to_numeric(data['delta_time'].dt.days, downcast='integer')
    data['event_month'] = data['DATE'].dt.month

    
    data['MAPPED_EVENT']=''

    map_international_events(data) # call function to map relevant events

 #   process_results(data) # convert results into seconds format

    return data

#@st.cache_data(ttl=20000)
#def fetch_all_data():  # fetch athlete results
    
#    all_data = client.query_and_wait(all_sql).to_dataframe()

#    all_data = clean_columns(all_data)  # clean name list of special characters, white spaces etc.

#    all_data['NAME'] = all_data['NAME'].str.casefold()

#    names['VARIATION'] = names['VARIATION'].str.casefold()
#    names['NAME'] = names['NAME'].str.casefold()

#    for row in names.itertuples():  # itertuples is faster
        
#        all_data['NAME'] = all_data['NAME'].replace(regex=rf"{row.VARIATION}", value=f"{row.NAME}")   


#    return all_data


@st.cache_data(ttl=20000)
def fetch_all_data():   # for database access

    all_data = client.query_and_wait(all_sql).to_dataframe()
 #   all_data = clean_columns(all_data)

    all_data=simple_map_events(all_data)

    
    # Casefold for consistency
    all_data['NAME'] = all_data['NAME'].str.casefold()

    # Standardize DATE column → always YYYY-MM-DD string
    all_data['DATE'] = pd.to_datetime(all_data['DATE'], errors='coerce') # NEW

    if pd.api.types.is_datetime64tz_dtype(all_data['DATE']): #NEW
        all_data['DATE'] = all_data['DATE'].dt.tz_localize(None)  #NEW

    # Convert to YYYY-MM-DD string
    all_data['DATE'] = all_data['DATE'].dt.strftime("%Y-%m-%d")
    
    # Work on a copy of names
    n = names[['VARIATION', 'NAME']].dropna().copy()
    n['VARIATION'] = n['VARIATION'].str.strip().str.casefold()

    # Remove ^ and $ anchors
    n['VARIATION'] = n['VARIATION'].str.replace(r'^\^', '', regex=True)
    n['VARIATION'] = n['VARIATION'].str.replace(r'\$$', '', regex=True)

    n['NAME'] = n['NAME'].str.casefold()

    # Drop duplicates → required for mapping
    mapping = n.drop_duplicates(subset='VARIATION').set_index('VARIATION')['NAME']

    # Vectorized replacement
    all_data['NAME'] = all_data['NAME'].map(mapping).fillna(all_data['NAME'])


 #   all_data = all_data[['NAME', 'DATE', 'MAPPED_EVENT', 'COMPETITION', 'RESULT', 'WIND', 'HOST_CITY', 'AGE', 'GENDER', 'EVENT_CLASS', 'DOB']]


    # Define a filter for rows with convertible results
    invalid_results = {'—', 'None', 'DQ', 'SCR', 'FS', 'DNQ', 'DNS', 'NH', 'NM', 'FOUL', 'DNF', 'SR'}

# Apply conversion vectorized using apply, skipping invalid values
    def convert_for_row(row):
        if row['RESULT'] in invalid_results:
            return ''
        return convert_time_refactored(row.name, row['MAPPED_EVENT'], row['RESULT'])

    all_data['RESULT_CONV'] = all_data.apply(convert_for_row, axis=1)
    
    return all_data

## Get all the data ##

all_data = fetch_all_data() # fetch the entire database
#data = fetch_data() # fetch the database of results for selected period


benchmark_option = st.selectbox(
    "  ",
    ("Search Database Records by Name or Competition", "List Results By Event", "2023 SEAG Bronze - SEAG Selection", 
     "2023 SEAG Bronze - OCTC Selection")
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

    
        
        all_data['NAME'] = all_data['NAME'].str.title()  # capitalize first letter (NEW)

        all_data['NAME_case'] = all_data["NAME"].str.casefold()

        combinations = all_data[all_data['NAME'].str.casefold().str.contains(text)]['NAME'].unique().tolist()
 
        #  name_selected = st.multiselect('Select From List of Matches :', all_data.loc[all_data['NAME_case'].str.contains(text)]['NAME'].unique())
        name_selected = st.selectbox('Select From List of Matches :', combinations)

     #   all_data['DATE'] = pd.to_datetime(all_data['DATE'], errors='coerce') # convert date column so mitosheet can search on dates  # MOVED
   
     #   if pd.api.types.is_datetime64tz_dtype(all_data['DATE']):  # remove tz awareness
     #       all_data['DATE'] = all_data['DATE'].dt.tz_localize(None)   # MOVED

        
        try:
      #  st.write(name_selected[0])
            m1 = all_data["NAME"].str.contains(name_selected)
        except:
            st.write("Keyword Does Not Exist in Database")
            m1 = all_data["NAME"].notnull()

        
        #all_data.loc[all_data['NAME_case'].str.contains(text)]['NAME_case'].unique()
        df_search = all_data[m1].sort_values(by='DATE', ascending=False)

        df_search = df_search[['NAME', 'DATE', 'MAPPED_EVENT', 'COMPETITION', 'RESULT', 'WIND', 'HOST_CITY', 'AGE', 'GENDER', 'EVENT_CLASS', 'DOB']]
## NEW BLOCK ##
        def seconds_to_mmss(seconds):
    #Converts total seconds (float) into a standardized time string format: HH:MM:SS.ss
            if pd.isna(seconds):
                return ''
    
    # 1. Calculate Hours, Minutes, and remaining Seconds
            hours, remainder = divmod(seconds, 3600)
            minutes, secs = divmod(remainder, 60)
    
            hours = int(hours)
            minutes = int(minutes)
    
            return f"{hours:02d}:{minutes:02d}:{secs:05.2f}"    

        distance_events = ['100m', '200m', '400m', '800m', '10,000m', '5000m', '3000m Steeplechase', '1500m', '10000m Racewalk', '1 Mile']
        field_events = ['Javelin Throw', 'Pole Vault', 'Hammer Throw', 'Triple Jump', 'Long Jump', 'High Jump', 'Shot Put', 'Discus Throw']

        invalid_results = {'—', 'None', 'DQ', 'SCR', 'FS', 'DNQ', 'DNS', 'NH', 'NM', 'FOUL', 'DNF', 'SR', '', ' '}

        def convert_for_row(row):
            if row['RESULT'] in invalid_results:
                return ''
            return convert_time_refactored(row.name, row['MAPPED_EVENT'], row['RESULT'])

  
   # def convert_for_row(row):
   #     if row['RESULT'] in invalid_results:
   #         return np.nan
   #     result = convert_time_refactored(row.name, row['EVENT'], row['RESULT'])
   #     if result == '' or result is None:
   #         st.write(f"FAILED PARSE: EVENT={row['EVENT']} RESULT={row['RESULT']}")
   #         return np.nan
   #     return result

    # 1. Create Mask for Results containing 'w' (illegal wind speed indicator)
        mask_result_has_w = athletes['RESULT'].astype(str).str.contains('w', case=False, na=False)

    # 2. Create Mask for Missing/Empty Wind Field
    # This robustly captures NaN, empty string (''), and strings containing only whitespace (' ')
        mask_wind_is_missing = (
        athletes['WIND'].isna() 
        | (athletes['WIND'].astype(str).str.strip() == '')
        | (athletes['WIND'].astype(str).str.lower().str.strip().isin(['nan', 'none', '-']))
        )

    # 3. Combine the masks: Only update the WIND field if the result has 'w' AND the WIND field is missing.
        final_mask = mask_result_has_w & mask_wind_is_missing

    # 4. Apply the mask: Set the 'WIND' field to 'Illegal'
        athletes.loc[final_mask, 'WIND'] = 'Illegal'

        df_search['RESULT_FLOAT'] = df_search.apply(convert_for_row, axis=1)

        st.dataframe(df_search)

        df_search['RESULT_FLOAT'] = pd.to_numeric(df_search['RESULT_FLOAT'], errors='coerce')
    
        df_search['RESULT_FLOAT'] = df_search['RESULT_FLOAT'].replace('', np.nan)


      #  df_search = df_search[df_search['RESULT_FLOAT'].notna()]  # UNCOMMENT THIS IF REQUIRED


# 2. Create the boolean mask using .isin()
# Assuming the column you are checking is called 'EVENT_TYPE' (replace if different)
        mask = df_search['MAPPED_EVENT'].isin(distance_events)
        mask_field = df_search['MAPPED_EVENT'].isin(field_events)


# 3. Apply the 'seconds_to_mmss' function ONLY to the masked rows
# This replaces the original .apply() within the if block.
        df_search.loc[mask, 'RESULT_C'] = (df_search.loc[mask, 'RESULT_FLOAT'].apply(seconds_to_mmss))
        df_search.loc[mask_field, 'RESULT_C'] = (df_search.loc[mask_field, 'RESULT_FLOAT'])


# 4. Convert the new column to timedelta
# This operation is already vectorised (applied to the whole column/Series at once).
  #      df_search['timedelta'] = pd.to_timedelta(df_search['RESULT_TIMES'])


    ## END NEW BLOCK ##      

        
        if text_search:
      #      st.write(df_search)
            final_dfs, code = spreadsheet(df_search)

        all_data.drop(['NAME_case'], axis=1, inplace=True)

        
    

    elif search_option=='Competition Name':

        text_search = st.text_input("Enter Search Keyword for Competition Name", value="")
        text = text_search.casefold()


        all_data['COMPETITION_case'] = all_data["COMPETITION"].str.casefold()
        
        try:
      #  st.write(name_selected[0])
            m2 = all_data["COMPETITION_case"].str.contains(text)
        except:
            st.write("Keyword Does Not Exist in Database")
            m2 = all_data["COMPETITION_case"].notnull()

        all_data['DATE'] = pd.to_datetime(all_data['DATE'], errors='coerce') # convert date column so mitosheet can search on dates

        if pd.api.types.is_datetime64tz_dtype(all_data['DATE']):  # remove timezone awareness
            all_data['DATE'] = all_data['DATE'].dt.tz_convert(None)

        df_search = all_data[m2]

  #      df_search = df_search[['NAME', 'TEAM', 'RESULT', 'WIND', 'EVENT', 'DIVISION', 'STAGE', 'AGE', 'GENDER', 'NATIONALITY', 'DICT_RESULTS', 'DATE', 'COMPETITION', 'DOB',
  #                      'REGION', 'REMARKS', 'SUB_EVENT', 'DISTANCE']]
    
        if text_search:
        #    st.write(df_search)
            final_dfs, code = spreadsheet(df_search)

        all_data.drop(['COMPETITION_case'], axis=1, inplace=True)

## List Results BY Event##

elif benchmark_option == 'List Results By Event':

    events_list = all_data['EVENT'].str.casefold().unique().tolist()

    list_option = st.selectbox(
    "Select Event:",
    options = events_list,
    )

    searched_event = all_data[all_data['EVENT'].str.casefold()==list_option]

    searched_event = clean_columns(searched_event)

    searched_event['DATE'] = pd.to_datetime(searched_event['DATE'], format='%Y-%m-%d')

    invalid_results = {'—', 'None', 'DQ', 'SCR', 'FS', 'DNQ', 'DNS', 'NH', 'NM', 'FOUL', 'DNF', 'SR', '', ' '}

    def convert_for_row(row):
        if row['RESULT'] in invalid_results:
            return ''
        return convert_time_refactored(row.name, row['EVENT'], row['RESULT'])

  
   # def convert_for_row(row):
   #     if row['RESULT'] in invalid_results:
   #         return np.nan
   #     result = convert_time_refactored(row.name, row['EVENT'], row['RESULT'])
   #     if result == '' or result is None:
   #         st.write(f"FAILED PARSE: EVENT={row['EVENT']} RESULT={row['RESULT']}")
   #         return np.nan
   #     return result


    searched_event['RESULT_FLOAT'] = searched_event.apply(convert_for_row, axis=1)
    
    searched_event['RESULT_FLOAT'] = searched_event['RESULT_FLOAT'].replace('', np.nan)

    searched_event = searched_event[searched_event['RESULT_FLOAT'].notna()]
 #   searched_event = searched_event.sort_values(by='RESULT_FLOAT', ascending=True, na_position='last')


    #def seconds_to_mmss(seconds):
    #    if pd.isna(seconds):
    #        return ''
    #    minutes, secs = divmod(seconds, 60)
    #    return f"{int(minutes):02d}:{secs:05.2f}"
    def seconds_to_mmss(seconds):
    #Converts total seconds (float) into a standardized time string format: HH:MM:SS.ss
        if pd.isna(seconds):
            return ''
    
    # 1. Calculate Hours, Minutes, and remaining Seconds
        hours, remainder = divmod(seconds, 3600)
        minutes, secs = divmod(remainder, 60)
    
    # Ensure all components are integers for clean formatting, except for secs
        hours = int(hours)
        minutes = int(minutes)
    
    # 2. Format the time string: HH:MM:SS.ss
    # :02d ensures two digits with a leading zero (e.g., 00, 09, 10)
    # :05.2f ensures two decimal places and pads with leading zeros if needed
    # for a total width of 5 (e.g., 5.29 -> 05.29 is not guaranteed, but 5.29 is consistent)
    # For secs, we just want the two decimal places and minimal padding.
    
        return f"{hours:02d}:{minutes:02d}:{secs:05.2f}"    


   
    if list_option=='800m' or list_option=='10,000m' or list_option=='5000m' or list_option=='3000m steeplechase' or list_option=='1500m' or list_option=='10000m racewalk':
   
    
        searched_event['RESULT_TIMES'] = searched_event['RESULT_FLOAT'].apply(seconds_to_mmss)  

        searched_event['timedelta'] = pd.to_timedelta(searched_event['RESULT_TIMES']) # Convert to timedelta format



    final_dfs, code = spreadsheet(searched_event)

    # Show the results, if you have a text_search
    
    
#    final_dfs, code = spreadsheet(all_data)
  #  st.write(final_dfs)

    # Allow text search on athlete name and/or competition
    
    
else:  # Choose date and run selection report

    data = fetch_data() # fetch the database of results for selected period


    # Choose start and end dates
    
    if benchmark_option == '2023 SEAG Bronze - SEAG Selection':
      
        start = '2024-10-22'
        end = '2025-09-05'
        start_date = pd.to_datetime(start)
        end_date = pd.to_datetime(end)

    
    elif benchmark_option == '2023 SEAG Bronze - OCTC Selection':
       # start = np.datetime64(start_date)
       # end = np.datetime64(end_date)
        start = '2024-01-01'
        end = '2025-12-31'
        start_date = pd.to_datetime(start)
        end_date = pd.to_datetime(end)


    
    data['DATE'] = pd.to_datetime(data['DATE'], format='mixed', dayfirst=False, utc=True)
    data['DATE'] = data['DATE'].dt.tz_localize(None)  # switch off timezone for compatibility with np.datetime64
    

    mask = ((data['DATE'] >= start) & (data['DATE'] <= end))
    athletes_selected = data.loc[mask]
    athletes_selected.reset_index(drop=True, inplace=True)

    # Fast lookup for benchmarks
  #  bench_map = {
  #      "2023 SEAG Bronze - SEAG Selection": '2023 SEAG Bronze',
  ##      "2023 SEAG Bronze - OCTC Selection": '2023 SEAG Bronze',
   #     "26th Asian Athletics": '26th Asian Athletics',
   #     "2025 Taiwan Open": '2025 Taiwan Open'
   # }

    bench_map = {
        "2023 SEAG Bronze - SEAG Selection": '2023 SEAG Bronze',
        "2023 SEAG Bronze - OCTC Selection": '2023 SEAG Bronze'
            }
    
    bench_name = bench_map.get(benchmark_option, None)
    benchmark = benchmarks[benchmarks['BENCHMARK_COMPETITION'] == bench_name] if bench_name else pd.DataFrame()


## Map relevant events to a standard description ##

## Map benchmarks ##

#st.dataframe(athletes_selected)

## Merge benchmarks ##

if benchmark_option == '2023 SEAG Bronze - SEAG Selection' or benchmark_option == '2023 SEAG Bronze - OCTC Selection':

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

    

# Iterate over dataframe and replace names using casefold then convert to capitalize first letter (OLD BLOCK)

#    df['NAME'] = df['NAME'].str.casefold()  # convert everything to lower case (NEW)
    
#    names['VARIATION'] = names['VARIATION'].str.casefold()
#    names['NAME'] = names['NAME'].str.casefold()

#    for row in names.itertuples():  # itertuples is faster
        
#        df['NAME'] = df['NAME'].replace(regex=rf"{row.VARIATION}", value=f"{row.NAME}")   
#        df['NAME'] = df['NAME'].replace(regex=pattern, value=f"{row.NAME}")   

#    df['NAME'] = df['NAME'].str.title()  # capitalize first letter (NEW)

# END OLD BLOCK

# Iterate over dataframe and replace names using casefold then convert to capitalize first letter (NEW BLOCK)
# Normalize dataframe

    df['NAME'] = df['NAME'].apply(normalize_text)
    names['VARIATION'] = names['VARIATION'].apply(normalize_text)
    names['NAME'] = names['NAME'].apply(normalize_text)

# Precompile all regex patterns safely

    compiled_patterns = []
    for pattern_str, replacement in zip(names['VARIATION'], names['NAME']):
        try:
            compiled_re = re.compile(pattern_str)
            compiled_patterns.append( (compiled_re, replacement) )
        except re.error as e:
            print(f"Skipping invalid regex pattern: {pattern_str} Error: {e}")

# Iterate over all patterns and apply replacements using precompiled regexes
    for regex, replacement in compiled_patterns:
        df['NAME'] = df['NAME'].str.replace(regex, replacement, regex=True)

# Capitalize final standardized names
    df['NAME'] = df['NAME'].str.title()

# END NEW BLOCK #


        

    
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

##        all_ranking_octc = df_no_na.sort_values(['MAPPED_EVENT','GENDER','PERF_SCALAR'], ascending=[False, False, False])
##        all_ranking_octc['Rank'] = all_ranking_octc.groupby(['GENDER', 'MAPPED_EVENT', 'TIER']).cumcount() + 1

##        all_ranking_octc['TIER_ADJ'] = np.where(
##                                ((all_ranking_octc['TIER']=='Tier 1') & (all_ranking_octc['Rank']==3)), 'Tier 2',    
##                                np.where(
##                                ((all_ranking_octc['TIER']=='Tier 1') & (all_ranking_octc['Rank']>=4)), 'Tier 2',
##                                np.where(
##                                ((all_ranking_octc['TIER']=='Tier 2') & (all_ranking_octc['Rank']==3)), 'Tier 3', 
##                                np.where(
##                                ((all_ranking_octc['TIER']=='Tier 2') & (all_ranking_octc['Rank']>=4)), 'Tier 3', 
##                                np.where(                             
##                                ((all_ranking_octc['TIER']=='Tier 3') & (all_ranking_octc['Rank']==3)), 'Tier 4', 
##                                np.where(                             
##                                ((all_ranking_octc['TIER']=='Tier 3') & (all_ranking_octc['Rank']>=4)), 'Tier 4', all_ranking_octc['TIER']) 
                
##                                )))))

        

##        rerank_octc = all_ranking_octc.sort_values(['MAPPED_EVENT','GENDER','TIER_ADJ', 'PERF_SCALAR'], ascending=[False, False, False, False])
##        rerank_octc['Rank_ADJ'] = rerank_octc.groupby(['MAPPED_EVENT', 'GENDER', 'TIER_ADJ']).cumcount() + 1

##        rerank_filtered_octc = rerank_octc[(rerank_octc['TIER_ADJ']!=' ') & (rerank_octc['TIER_ADJ']!='Tier 4')]

##        rerank_filtered_octc = rerank_filtered_octc.drop(['TIER', 'Rank'], axis=1)

##        rerank_filtered_octc.rename(columns={'TIER_ADJ': 'TIER', 'Rank_ADJ': 'TIER_RANKING'}, inplace=True)
## NEW BLOCK
        # 1. Sort and Rank
            all_ranking_octc = df_no_na.sort_values(['MAPPED_EVENT', 'GENDER', 'PERF_SCALAR'], ascending=[False, False, False])
            all_ranking_octc['Rank'] = all_ranking_octc.groupby(['GENDER', 'MAPPED_EVENT', 'TIER']).cumcount() + 1

        # 2. Define Tier Adjustment Logic
            def adjust_tier(row):
                tier, rank = row['TIER'], row['Rank']
                if tier == 'Tier 1' and rank >= 3:
                    return 'Tier 2'
                elif tier == 'Tier 2' and rank >= 3:
                    return 'Tier 3'
                elif tier == 'Tier 3' and rank >= 3:
                    return 'Tier 4'
                else:
                    return tier

        # Apply Tier Adjustment
            all_ranking_octc['TIER_ADJ'] = all_ranking_octc.apply(adjust_tier, axis=1)

        # 3. Secondary Sort and Re-Rank
            rerank_octc = all_ranking_octc.sort_values(['MAPPED_EVENT', 'GENDER', 'TIER_ADJ', 'PERF_SCALAR'], ascending=[False, False, False, False])
            rerank_octc['Rank_ADJ'] = rerank_octc.groupby(['MAPPED_EVENT', 'GENDER', 'TIER_ADJ']).cumcount() + 1

        # 4. Filter, Clean, and Rename
            rerank_filtered_octc = rerank_octc[~rerank_octc['TIER_ADJ'].isin([' ', 'Tier 4'])].drop(['TIER', 'Rank'], axis=1)
            rerank_filtered_octc.rename(columns={'TIER_ADJ': 'TIER', 'Rank_ADJ': 'TIER_RANKING'}, inplace=True)

    ## NEW BLOCK END##
        
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


    
  #  st.write(final_df)

    final_dfs, code = spreadsheet(final_df)



