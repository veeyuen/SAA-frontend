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

## DEFINE FUNCTONS ##

# Converts any time format into seconds

def convert_time(i, string, metric):

    global output
    
    l=['discus', 'throw', 'jump', 'vault', 'shot']
        
    string=string.lower()

    output=''
    
   # print('metric', metric)
    
    try:
        
        if 'w' in metric:  # skip marks with illegal wind speeds
            
            print('W', metric)
            
            output=''
            
        else:
            
    
            if any(s in string for s in l)==True:
            
                if 'm' in metric:
            
                    metric=metric.replace('m', '')
                    output=float(str(metric))
            
                elif 'GR' in metric:
            
                    metric=metric.replace('GR', '')
                    output=float(str(metric))
                
                
                else:
    
                    output=float(str(metric))
        
                
                
        
            else:
        
                searchstring = ":"
                searchstring2 = "."
                substring=str(metric)
                count = substring.count(searchstring)
                count2 = substring.count(searchstring2)
            
                if count==0:
                
                    output=float(substring)
            
            
               
             
                elif (type(metric)==datetime.time or type(metric)==datetime.datetime):
                
                                                
                    time=str(metric)
                    h, m ,s = time.split(':')
                    output = float(datetime.timedelta(hours=int(h),minutes=int(m),seconds=float(s)).total_seconds())
            
                                
                elif (count==1 and count2==1):
            
                    m,s = metric.split(':')
                    output = float(datetime.timedelta(minutes=int(m),seconds=float(s)).total_seconds())
                     
                elif (count==1 and count2==2):
                
            
                    metric = metric.replace(".", ":", 1)
            
                    h,m,s = metric.split(':')            
                    output = float(datetime.timedelta(hours=int(h),minutes=int(m),seconds=float(s)).total_seconds())
                
        
                elif (count==2 and count2==0):
                
            
                    h,m,s = metric.split(':')
                    output = float(datetime.timedelta(hours=int(h),minutes=int(m),seconds=float(s)).total_seconds())
  
            

    except:
        
        pass
                
    return output
    
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



#
#df = client.query_and_wait(all_sql).to_dataframe()

#df.dropna(how= "all", axis=1, inplace=True)

#year_list = df['YEAR'].unique().tolist() # get unique list of years
#region_list = df['REGION'].unique().tolist()
#competition_list = df['COMPETITION'].unique().tolist()

#year_selection = st.multiselect(
#    "Please select the desired year(s):",
#    year_list,
#)

#region_selection = st.multiselect(
#    "Please select the desired region(S):",
#    region_list,
#)

#competition_selection = st.multiselect(
#    "Please select the desired competition(s):",
#    competition_list,
#)


#df_filtered = df[df['DATE'].isin(year_selection) & df['REGION'].isin(region_selection) & df['COMPETITION'].isin(competition_selection)]


### EXTRACT LIST OF ATHLETES ###

#athletes = client.query_and_wait(athletes_sql).to_dataframe()

# SELECT YEARS

selection = client.query_and_wait(all_sql).to_dataframe()

selection.dropna(how= "all", axis=1, inplace=True)

year_list = selection['YEAR'].unique().tolist() # get unique list of years
#region_list = df['REGION'].unique().tolist()
#competition_list = df['COMPETITION'].unique().tolist()

year_selection = st.multiselect(
    "Please select the desired year(s):",
    year_list,
)

athletes = selection[selection['YEAR'].isin(year_selection)] # filter results based on selected year


# Create temporary mapped event column

athletes['MAPPED_EVENT']=''

# Clear columns of special characters and spaces

for col in athletes.columns:
    athletes[col] = athletes[col].astype(str)
    athletes[col] = athletes[col].str.replace('\xa0', ' ', regex=True)
    athletes[col] = athletes[col].str.replace('[\x00-\x1f\x7f-\x9f]', '', regex=True)
    athletes[col] = athletes[col].str.replace('\r', ' ', regex=True)
    athletes[col] = athletes[col].str.replace('\n', ' ', regex=True)
    athletes[col] = athletes[col].str.strip()

# Correct javelin category

mask = athletes['EVENT'].str.contains(r'Javelin', na=True)
athletes.loc[mask, 'CATEGORY_EVENT'] = 'Throw'

# Map running categories

mask = (athletes['EVENT'].str.contains(r'Dash', na=True) & athletes['DISTANCE'].str.contains(r'100', na=True))
athletes.loc[mask, 'MAPPED_EVENT'] = '100m'
mask = (athletes['EVENT'].str.contains(r'Run', na=True) & athletes['DISTANCE'].str.contains(r'100', na=True))
athletes.loc[mask, 'MAPPED_EVENT'] = '100m'

mask = athletes['EVENT'].str.contains(r'100 Meter Run', na=True)
athletes.loc[mask, 'MAPPED_EVENT'] = '100m'
mask = athletes['EVENT'].str.contains(r'^100m$', na=True)
athletes.loc[mask, 'MAPPED_EVENT'] = '100m'
mask = (athletes['EVENT'].str.contains(r'Dash', na=True) & athletes['DISTANCE'].str.contains(r'200', na=True))
athletes.loc[mask, 'MAPPED_EVENT'] = '200m'
mask = (athletes['EVENT'].str.contains(r'Run', na=True) & athletes['DISTANCE'].str.contains(r'200', na=True))
athletes.loc[mask, 'MAPPED_EVENT'] = '200m'

mask = athletes['EVENT'].str.contains(r'^200m$', na=True)
athletes.loc[mask, 'MAPPED_EVENT'] = '200m'
mask = athletes['EVENT'].str.contains(r'200\sMeter', na=True)
athletes.loc[mask, 'MAPPED_EVENT'] = '200m'
mask = athletes['EVENT'].str.contains(r'300 Meter Run', na=True)
athletes.loc[mask, 'MAPPED_EVENT'] = '300m'
mask = (athletes['EVENT'].str.contains(r'Dash', na=True) & athletes['DISTANCE'].str.contains(r'400', na=True))
athletes.loc[mask, 'MAPPED_EVENT'] = '400m'
mask = athletes['EVENT'].str.contains(r'^400m$', na=True)
athletes.loc[mask, 'MAPPED_EVENT'] = '400m'

mask = athletes['EVENT'].str.contains(r'^400\sMeter$', na=True)
athletes.loc[mask, 'MAPPED_EVENT'] = '400m'


mask = (athletes['EVENT'].str.contains(r'Run', na=True) & athletes['DISTANCE'].str.contains(r'^600$', na=True, regex=True))
athletes.loc[mask, 'MAPPED_EVENT'] = '600m'
mask = (athletes['EVENT'].str.contains(r'Run', na=True) & athletes['DISTANCE'].str.contains(r'800', na=True))
athletes.loc[mask, 'MAPPED_EVENT'] = '800m'
mask = athletes['EVENT'].str.contains(r'800 Meter Run', na=True)
athletes.loc[mask, 'MAPPED_EVENT'] = '800m'
mask = athletes['EVENT'].str.contains(r'^800m$', na=True)
athletes.loc[mask, 'MAPPED_EVENT'] = '800m'
mask = (athletes['EVENT'].str.contains(r'Run', na=True) & athletes['DISTANCE'].str.contains(r'1000', na=True))
athletes.loc[mask, 'MAPPED_EVENT'] = '1000m'


mask = (athletes['EVENT'].str.contains(r'Run', na=True) & athletes['DISTANCE'].str.contains(r'1500', na=True))
athletes.loc[mask, 'MAPPED_EVENT'] = '1500m'
mask = athletes['EVENT'].str.contains(r'^1500m$', na=True, regex=True)
athletes.loc[mask, 'MAPPED_EVENT'] = '1500m'
mask = (athletes['EVENT'].str.contains(r'Run', na=True) & athletes['DISTANCE'].str.contains(r'3000', na=True))
athletes.loc[mask, 'MAPPED_EVENT'] = '3000m'
mask = (athletes['EVENT'].str.contains(r'Run', na=True) & athletes['DISTANCE'].str.contains(r'5000', na=True))
athletes.loc[mask, 'MAPPED_EVENT'] = '5000m'
mask = athletes['EVENT'].str.contains(r'^5000m$', na=True)
athletes.loc[mask, 'MAPPED_EVENT'] = '5000m'
mask = (athletes['EVENT'].str.contains(r'Run', na=True) & athletes['DISTANCE'].str.contains(r'10000', na=True))
athletes.loc[mask, 'MAPPED_EVENT'] = '10,000m'
mask = athletes['EVENT'].str.contains(r'^10000m$', na=True)
athletes.loc[mask, 'MAPPED_EVENT'] = '10,000m'
mask = (athletes['EVENT'].str.contains(r'Run', na=True) & athletes['DISTANCE'].str.contains(r'Mile', na=True))
athletes.loc[mask, 'MAPPED_EVENT'] = '1 Mile'

# Map hurdles

mask = (athletes['EVENT'].str.contains(r'100m Hurdles|100m hurdles', na=False) & athletes['EVENT_CLASS'].str.contains('0.84', na=False) & athletes['GENDER'].str.contains(r'Female', na=False))  # this is the correct syntax
athletes.loc[mask, 'MAPPED_EVENT'] = '100m Hurdles'
mask = (athletes['EVENT'].str.contains(r'100m Hurdles|100m hurdles', na=False) & athletes['DIVISION'].str.contains('None', na=False) & athletes['GENDER'].str.contains(r'Female', na=False) & athletes['REGION'].str.contains(r'International', na=False))  # this is the correct syntax
athletes.loc[mask, 'MAPPED_EVENT'] = '100m Hurdles'
mask = (athletes['EVENT'].str.contains(r'^Hurdles$', na=False) & athletes['DISTANCE'].str.contains(r'100', na=False) & athletes['DIVISION'].str.contains(r'OPEN|Open', na=False) & athletes['GENDER'].str.contains(r'Female', na=False))
athletes.loc[mask, 'MAPPED_EVENT'] = '100m Hurdles'

mask = (athletes['EVENT'].str.contains(r'^Hurdles$', na=False) & athletes['DISTANCE'].str.contains(r'110', na=False) & athletes['DIVISION'].str.contains(r'OPEN|Open', na=False) & athletes['GENDER'].str.contains(r'Male', na=False))
athletes.loc[mask, 'MAPPED_EVENT'] = '110m Hurdles'
mask = (athletes['EVENT'].str.contains(r'^Hurdles$', na=False) & athletes['DISTANCE'].str.contains(r'110', na=False) & athletes['EVENT_CLASS'].str.contains(r'0.838|0.84', na=False) & athletes['GENDER'].str.contains(r'Female', na=False))
athletes.loc[mask, 'MAPPED_EVENT'] = '110m Hurdles'
mask = ((athletes['EVENT'].str.contains(r'110m Hurdles|110m hurdles', na=False)) 
         & ((athletes['EVENT_CLASS'].str.contains('None', na=False))|(athletes['EVENT_CLASS']==np.nan)|(athletes['EVENT_CLASS']=='')) 
         & athletes['REGION'].str.contains(r'International', na=False) & (athletes['DIVISION'].str.contains(r'None', na=False)))  # this is the correct syntax
athletes.loc[mask, 'MAPPED_EVENT'] = '110m Hurdles'
                                

mask = (athletes['EVENT'].str.contains(r'^Hurdles$', na=False) & athletes['DISTANCE'].str.contains(r'110', na=False) & athletes['EVENT_CLASS'].str.contains(r'1.067', na=False))
athletes.loc[mask, 'MAPPED_EVENT'] = '110m Hurdles'
mask = (athletes['EVENT'].str.contains(r'^Hurdles$', na=False) & athletes['DISTANCE'].str.contains(r'110', na=False) & athletes['EVENT_CLASS'].str.contains(r'0.914', na=False) & athletes['GENDER'].str.contains(r'Female', na=False))
athletes.loc[mask, 'MAPPED_EVENT'] = '110m Hurdles'

mask = (athletes['EVENT'].str.contains(r'^Hurdles$', na=False) & athletes['DISTANCE'].str.contains(r'400', na=False) & athletes['EVENT_CLASS'].str.contains(r'0.84|84cm', na=False) & athletes['GENDER'].str.contains(r'Female', na=False))
athletes.loc[mask, 'MAPPED_EVENT'] = '400m Hurdles'


mask = (athletes['EVENT'].str.contains(r'^Hurdles$', na=False) & athletes['DISTANCE'].str.contains(r'400', na=False) & athletes['EVENT_CLASS'].str.contains(r'0.914', na=False))
athletes.loc[mask, 'MAPPED_EVENT'] = '400m Hurdles'
mask = (athletes['EVENT'].str.contains(r'^Hurdles$', na=False) & athletes['DISTANCE'].str.contains(r'400', na=False) & athletes['DIVISION'].str.contains(r'Open|Invitational', na=False))
athletes.loc[mask, 'MAPPED_EVENT'] = '400m Hurdles'

mask = (athletes['EVENT'].str.contains(r'400m Hurdles', na=False) & athletes['EVENT_CLASS'].str.contains(r'0.914', na=False)  & athletes['GENDER'].str.contains(r'Male', na=False))
athletes.loc[mask, 'MAPPED_EVENT'] = '400m Hurdles'


mask = (athletes['EVENT'].str.contains(r'Hurdles', na=False) & athletes['DISTANCE'].str.contains(r'400', na=False) & athletes['EVENT_CLASS'].str.contains(r'0.762', na=False)& athletes['GENDER'].str.contains(r'Female', na=False))
athletes.loc[mask, 'MAPPED_EVENT'] = '400m Hurdles'
mask = (athletes['EVENT'].str.contains(r'400m Hurdles', na=False) & athletes['EVENT_CLASS'].str.contains(r'0.762m', na=False) & athletes['GENDER'].str.contains(r'Female', na=False))
athletes.loc[mask, 'MAPPED_EVENT'] = '400m Hurdles'
mask = (athletes['EVENT'].str.contains(r'400m Hurdles|400m hurdles', na=False) & athletes['EVENT_CLASS'].str.contains('None|0.762|0.914', na=False) & athletes['REGION'].str.contains(r'International', na=False))  # this is the correct syntax
athletes.loc[mask, 'MAPPED_EVENT'] = '400m Hurdles'


# Throws

mask = (athletes['EVENT'].str.contains(r'Javelin Throw|Javelin throw|Javelin', na=False) & athletes['EVENT_CLASS'].str.contains(r'600g', na=False) & athletes['GENDER'].str.contains(r'Female', na=False))
athletes.loc[mask, 'MAPPED_EVENT'] = 'Javelin Throw'
mask = (athletes['EVENT'].str.contains(r'Javelin Throw|Javelin throw|Javelin', na=False) & athletes['EVENT_CLASS'].str.contains(r'800g', na=False) & athletes['GENDER'].str.contains(r'Male', na=False))
athletes.loc[mask, 'MAPPED_EVENT'] = 'Javelin Throw'
mask = (athletes['EVENT'].str.contains(r'Javelin Throw|Javelin throw', na=False) & athletes['DIVISION'].str.contains(r'OPEN|Open', na=False))
athletes.loc[mask, 'MAPPED_EVENT'] = 'Javelin Throw'

mask = (athletes['EVENT'].str.contains(r'Shot Put|Shot put', na=False, regex=True) & (athletes['GENDER']=='Female') & (athletes['EVENT_CLASS']=='4kg'))# there are some additional characters after Put
athletes.loc[mask, 'MAPPED_EVENT'] = 'Shot Put'

mask = (athletes['EVENT'].str.contains(r'Shot Put|Shot put', na=False) & (athletes['GENDER']=='Male') & (athletes['EVENT_CLASS'].str.contains(r'7.26', na=False)))# there are some additional characters after Put
athletes.loc[mask, 'MAPPED_EVENT'] = 'Shot Put'
mask = (athletes['EVENT'].str.contains(r'Shot Put|Shot put', na=False) & (athletes['GENDER']=='Female') & (athletes['EVENT_CLASS'].str.contains(r'4', na=False)))# there are some additional characters after Put
athletes.loc[mask, 'MAPPED_EVENT'] = 'Shot Put'

mask = (athletes['EVENT'].str.contains(r'Shot Put|Shot put', na=False) & (athletes['DIVISION'].str.contains(r'OPEN|Open', na=False)))# there are some additional characters after Put
athletes.loc[mask, 'MAPPED_EVENT'] = 'Shot Put'

mask = (athletes['EVENT'].str.contains(r'Hammer Throw|Hammer throw', na=False) & athletes['EVENT_CLASS'].str.contains(r'7.26kg', na=False))
athletes.loc[mask, 'MAPPED_EVENT'] = 'Hammer Throw'
mask = (athletes['EVENT'].str.contains(r'Hammer Throw|Hammer throw', na=False) & athletes['EVENT_CLASS'].str.contains(r'4.00kg', na=False))
athletes.loc[mask, 'MAPPED_EVENT'] = 'Hammer Throw'
mask = (athletes['EVENT'].str.contains(r'Hammer Throw|Hammer throw', na=False) & (athletes['DIVISION'].str.contains(r'OPEN|Open', na=False)))# there are some additional characters after Put
athletes.loc[mask, 'MAPPED_EVENT'] = 'Hammer Throw'

mask = (athletes['EVENT'].str.contains(r'Discus Throw|Discus|Discus throw', na=False) & athletes['EVENT_CLASS'].str.contains(r'2kg|2.00kg', na=False) & athletes['GENDER'].str.contains(r'Male', na=False))
athletes.loc[mask, 'MAPPED_EVENT'] = 'Discus Throw'
mask = (athletes['EVENT'].str.contains(r'Discus Throw|Discus|Discus throw', na=False) & athletes['EVENT_CLASS'].str.contains(r'1kg|1.00kg', na=False) & athletes['GENDER'].str.contains(r'Female', na=False))
athletes.loc[mask, 'MAPPED_EVENT'] = 'Discus Throw'

mask = (athletes['EVENT'].str.contains(r'Discus Throw|Discus throw', na=False) & athletes['DIVISION'].str.contains(r'OPEN|Open', na=False))
athletes.loc[mask, 'MAPPED_EVENT'] = 'Discus Throw'
mask = (athletes['EVENT'].str.contains(r'Discus Throw|Discus throw', na=False) & athletes['DIVISION'].str.contains(r'None', na=False) & athletes['EVENT_CLASS'].str.contains(r'None', na=False))
athletes.loc[mask, 'MAPPED_EVENT'] = 'Discus Throw'


# Jumps

mask = athletes['EVENT'].str.contains(r'High Jump', na=True)
athletes.loc[mask, 'MAPPED_EVENT'] = 'High Jump'

mask = athletes['EVENT'].str.contains(r'^Long\sJump$', na=True)
athletes.loc[mask, 'MAPPED_EVENT'] = 'Long Jump'
mask = athletes['EVENT'].str.contains(r'Long Jump Open', na=True)
athletes.loc[mask, 'MAPPED_EVENT'] = 'Long Jump'
mask = athletes['EVENT'].str.contains(r'Long Jump Trial', na=True)
athletes.loc[mask, 'MAPPED_EVENT'] = 'Long Jump'


mask = athletes['EVENT'].str.contains(r'Triple Jump', na=True)
athletes.loc[mask, 'MAPPED_EVENT'] = 'Triple Jump'
mask = athletes['EVENT'].str.contains(r'Pole Vault', na=True)
athletes.loc[mask, 'MAPPED_EVENT'] = 'Pole Vault'
mask = athletes['EVENT'].str.contains(r'High jump', na=True)
athletes.loc[mask, 'MAPPED_EVENT'] = 'High Jump'
mask = athletes['EVENT'].str.contains(r'Long jump', na=True)
athletes.loc[mask, 'MAPPED_EVENT'] = 'Long Jump'
mask = athletes['EVENT'].str.contains(r'Triple jump', na=True)
athletes.loc[mask, 'MAPPED_EVENT'] = 'Triple Jump'
mask = athletes['EVENT'].str.contains(r'^Pole\svault$', na=True)
athletes.loc[mask, 'MAPPED_EVENT'] = 'Pole Vault'

# Steeplechase

mask = (athletes['EVENT'].str.contains(r'Steeplechase', na=False) & athletes['DISTANCE'].str.contains(r'3000', na=False)  & athletes['EVENT_CLASS'].str.contains(r'0.914', na=False))
athletes.loc[mask, 'MAPPED_EVENT'] = '3000m Steeplechase'
mask = (athletes['EVENT'].str.contains(r'Steeplechase', na=False) & athletes['DISTANCE'].str.contains(r'3000', na=False)  & athletes['DIVISION'].str.contains(r'OPEN|Open', na=False))
athletes.loc[mask, 'MAPPED_EVENT'] = '3000m Steeplechase'


# Walk

mask = (athletes['EVENT'].str.contains(r'Race Walk', na=False) & athletes['DISTANCE'].str.contains(r'10000', na=False))
athletes.loc[mask, 'MAPPED_EVENT'] = '10000m Racewalk'


# Relay

mask = athletes['EVENT'].str.contains(r'4x80m Relay', na=True)
athletes.loc[mask, 'MAPPED_EVENT'] = '4 x 80m'
mask = athletes['EVENT'].str.contains(r'^4\sx\s100m$', na=True)
athletes.loc[mask, 'MAPPED_EVENT'] = '4 x 100m'
mask = athletes['EVENT'].str.contains(r'4x100m Relay', na=True)
athletes.loc[mask, 'MAPPED_EVENT'] = '4 x 100m'
mask = athletes['EVENT'].str.contains(r'4 X 100m Relay', na=True)
athletes.loc[mask, 'MAPPED_EVENT'] = '4 x 100m'
mask = (athletes['EVENT'].str.contains(r'Relay', na=False) & athletes['DISTANCE'].str.contains(r'400', na=False))
athletes.loc[mask, 'MAPPED_EVENT'] = '4 x 100m'

mask = athletes['EVENT'].str.contains(r'4x400m Relay', na=True)
athletes.loc[mask, 'MAPPED_EVENT'] = '4 x 400m'
mask = athletes['EVENT'].str.contains(r'4 X 400m Relay', na=True)
athletes.loc[mask, 'MAPPED_EVENT'] = '4 x 400m'
mask = athletes['EVENT'].str.contains(r'4x100 Meter Relay', na=True)
athletes.loc[mask, 'MAPPED_EVENT'] = '4 x 100m'
mask = (athletes['EVENT'].str.contains(r'Relay', na=False) & athletes['DISTANCE'].str.contains(r'1600', na=False))
athletes.loc[mask, 'MAPPED_EVENT'] = '4 x 400m'
mask = athletes['EVENT'].str.contains(r'^4\sx\s400m$', na=True)
athletes.loc[mask, 'MAPPED_EVENT'] = '4 x 400m'

# Map decathlon/heptathlon

mask = athletes['EVENT'].str.contains(r'^Heptathlon$', na=True)
athletes.loc[mask, 'MAPPED_EVENT'] = 'Heptathlon'
mask = athletes['EVENT'].str.contains(r'^Decathlon$', na=True)
athletes.loc[mask, 'MAPPED_EVENT'] = 'Decathlon'
mask = athletes['EVENT'].str.contains(r'Heptathlon', na=True)
athletes.loc[mask, 'MAPPED_EVENT'] = 'Heptathlon'
mask = athletes['EVENT'].str.contains(r'Decathlon', na=True)
athletes.loc[mask, 'MAPPED_EVENT'] = 'Decathlon'


### PROCESS BENCHMARKS ###

comparisons = client.query_and_wait(benchmark_sql).to_dataframe()

competition_list = comparisons['COMPETITION'].unique().tolist()
competition_year_list = comparisons['YEAR'].unique().tolist()

benchmark_selection = st.multiselect(
    "Please select the desired benchmark competition:",
    competition_list,
)

benchmark_year_selection = st.multiselect(
    "Please select the desired benchmark year:",
    competition_year_list,
)


benchmarks = comparisons[comparisons['YEAR'].isin(benchmark_year_selection) & comparisons['COMPETITION'].isin(benchmark_selection)]


benchmarks=benchmarks[benchmarks['HEAT'].isnull() & benchmarks['SUB_EVENT'].isnull()]  # r

benchmarks.rename(columns = {'RESULT':'BENCHMARK'}, inplace = True)
benchmarks.drop(['YEAR', 'HEAT', 'NAME', 'RANK', 'CATEGORY_EVENT', 'COMPETITION', 'STAGE'], axis=1, inplace=True)

# convert times in benchmarks to standard format

benchmarks = benchmarks.reset_index(drop=True)




for i in range(len(benchmarks)):
        
    rowIndex = benchmarks.index[i]

    input_string=benchmarks.iloc[rowIndex,0]
    
    metric=benchmarks.iloc[rowIndex,3]
    
    if metric==None:
        continue
        
    out = convert_time(i, input_string, metric)
    
    print(rowIndex, input_string, out)
     
    benchmarks.loc[rowIndex, 'Metric'] = out



# Calculate benchmarks for timed and distance events separately

mask = benchmarks['EVENT'].str.contains(r'jump|throw|Pole|put|Jump|Throw|pole|Put', na=True)



# For distance events

st.write(benchmarks)



benchmarks.loc[mask, '2%']=benchmarks['Metric']*0.98
benchmarks.loc[mask, '3.5%']=benchmarks['Metric']*0.965
benchmarks.loc[mask, '5%']=benchmarks['Metric']*0.95

# For timed events

benchmarks.loc[~mask, '2%']=benchmarks['Metric']*1.02
benchmarks.loc[~mask, '3.5%']=benchmarks['Metric']*1.035
benchmarks.loc[~mask, '5%']=benchmarks['Metric']*1.05

# Merge benchmarks with df

benchmarks['MAPPED_EVENT']=benchmarks['EVENT'].str.strip()

df = athletes.reset_index().merge(benchmarks.reset_index(), on=['MAPPED_EVENT','GENDER'], how='left')
df['RESULT'] = df['RESULT'].replace(regex=r'–', value=np.nan)
df['RESULT'] = df['RESULT'].replace(regex=r'-', value=np.nan)


# Convert df results into seconds format

# Convert results and seed into seconds format

for i in range(len(df)):
    
    result_out=''
    
        
    rowIndex = df.index[i]

    input_string=df.iloc[rowIndex,6]    # event description
    
    metric=df.iloc[rowIndex,1] # result
    
    if metric=='—' or metric=='DQ' or metric=='SCR' or metric=='FS' or metric=='DNQ' or metric=='DNS' or metric=='NH' or metric=='NM' or metric=='FOUL' or metric=='DNF' or metric=='SR' :
        continue
    
    result_out = convert_time(i, input_string, metric)
#    print('line', i, input_string, metric, result_out)
         
    df.loc[rowIndex, 'RESULT_CONV'] = result_out



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

st.write(benchmarks.columns.tolist()) 



# Name corrections
# Read name variations from GCS name lists bucket (Still in beta)


df['NAME'] = df['NAME'].str.replace('\xa0', '', regex=True)
df['NAME'] = df['NAME'].str.replace('[\x00-\x1f\x7f-\x9f]', '', regex=True)
df['NAME'] = df['NAME'].str.replace('\r', '', regex=True)
df['NAME'] = df['NAME'].str.replace('\n', '', regex=True)
df['NAME'] = df['NAME'].str.strip()


# Read csv of name variations from GCS bucket

file_path = "gs://name_variations/name_variations.csv"
names = pd.read_csv(file_path,
                 sep=",")

# Iterate over dataframe and replace names

for index, row in names.iterrows():
        
    df['NAME'] = df['NAME'].replace(regex=rf"{row['VARIATION']}", value=f"{row['NAME']}")

# Read list of foreigners from GCS bucket

file_path = "gs://name_lists/List of Foreigners.csv"
foreigners = pd.read_csv(file_path,
                 sep=",",
                 encoding="unicode escape")

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

ex_foreigners = df_local_teams.loc[~df['NAME'].str.casefold().isin(exclusions)]  # ~ means NOT IN. DROP spex carded athletes

top_performers_clean = ex_foreigners.sort_values(['MAPPED_EVENT', 'NAME','PERF_SCALAR'],ascending=False).groupby(['MAPPED_EVENT', 'NAME']).head(1)

top_performers_clean.reset_index(inplace=True)


#spexed_list = top_performers.loc[~octc_df['NAME'].str.casefold().isin(spex_athletes_casefold)]  # ~ means NOT IN. DROP spex carded athletes

#spexed_list.sort_values(['MAPPED_EVENT', 'GENDER', 'PERF_SCALAR'], ascending=[True, True, False], inplace=True)
#spexed_list['overall_rank'] = 1
#spexed_list['overall_rank'] = spexed_list.groupby(['MAPPED_EVENT', 'GENDER'])['overall_rank'].cumsum()

#Apply OCTC selection rule: max 6 for 100m/400m and max 3 for all other events

#spexed_list=spexed_list[(((spexed_list['MAPPED_EVENT']=='400m')|(spexed_list['MAPPED_EVENT']=='100m'))&(spexed_list['overall_rank']<7))|(~((spexed_list['MAPPED_EVENT']=='400m')|(spexed_list['MAPPED_EVENT']=='100m'))&(spexed_list['overall_rank']<4))]

# Create performance tier column

top_performers_clean['TIER'] = np.where((top_performers_clean['Delta_Benchmark']>=0), 'Tier 1',    
                                np.where(((top_performers_clean['Delta_Benchmark']<0) & (top_performers_clean['Delta2']>=0)), 'Tier2',
                                np.where(((top_performers_clean['Delta2']<0) & (top_performers_clean['Delta3.5']>=0)), 'Tier3', 
                                np.where(((top_performers_clean['Delta3.5']<0) & (top_performers_clean['Delta5']>=0)), 'Tier4', ' '))))


# Drop rows without a corresponding benchmark

final_df = top_performers_clean[top_performers_clean['BENCHMARK'].notna()]

# Show resulting OCTC dataframe

st.write("LIST OF OCTC SELECTION ATHLETES:")

st.write(final_df)


# Process custom threshold benchmark
'''
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

'''
####### OLD CODE BELOW ###

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
