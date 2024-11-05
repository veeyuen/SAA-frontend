# streamlit_app.py

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
            
            if output==229.90:
                print(metric, m, s, output, 'here')

                     
        elif (count==1 and count2==2):
            
            metric = metric.replace(".", ":", 1)
            
            h,m,s = metric.split(':')            
            output = float(datetime.timedelta(hours=int(h),minutes=int(m),seconds=float(s)).total_seconds())
                
        
        elif (count==2 and count2==0):
            
            h,m,s = metric.split(':')
            output = float(datetime.timedelta(hours=int(h),minutes=int(m),seconds=float(s)).total_seconds())
                       
    return output


### DEFINE SQL QUERIES ###

benchmark_sql = """
SELECT NAME, RESULT, RANK, EVENT, CATEGORY_EVENT, GENDER, COMPETITION, STAGE
FROM `saa-analytics.results.saa_full`
WHERE STAGE='Final' AND COMPETITION='SEA Games' AND RANK='3'
"""

athletes_sql="""
SELECT NAME, RESULT, AGE, RANK AS COMPETITION_RANK, EVENT, DOB, COUNTRY, CATEGORY_EVENT, GENDER, COMPETITION, DATE
FROM `saa-analytics.results.saa_full` 
WHERE RESULT!='NM' AND RESULT!='-' AND RESULT!='FOUL' AND RANK!='DNS' AND RESULT!='DNS' AND RESULT!='DNF' AND RESULT!='DNQ' AND RESULT!='DQ' AND RESULT IS NOT NULL

all_sql="""SELECT * FROM `saa-analytics.results.saa_full`"""



#
#df = client.query_and_wait(all_sql).to_dataframe()

#df.dropna(how= "all", axis=1, inplace=True)

#year_list = df['DATE'].unique().tolist() # get unique list of source events
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

athletes = client.query_and_wait(athletes_sql).to_dataframe()

# Create temporary mapped event column

athletes['MAPPED_EVENT']=''

# Correct javelin category

mask = athletes['EVENT'].str.contains(r'Javelin', na=True)
athletes.loc[mask, 'CATEGORY_EVENT'] = 'Throw'

# Correct running categories

mask = athletes['EVENT'].str.contains(r'50 Meter Dash', na=True)
athletes.loc[mask, 'MAPPED_EVENT'] = '50m'
mask = athletes['EVENT'].str.contains(r'60 Meter Dash', na=True)
athletes.loc[mask, 'MAPPED_EVENT'] = '60m'
mask = athletes['EVENT'].str.contains(r'80 Meter Dash', na=True)
athletes.loc[mask, 'MAPPED_EVENT'] = '80m'
mask = athletes['EVENT'].str.contains(r'100 Meter Dash', na=True)
athletes.loc[mask, 'MAPPED_EVENT'] = '100m'
mask = athletes['EVENT'].str.contains(r'100 Meter Run', na=True)
athletes.loc[mask, 'MAPPED_EVENT'] = '100m'
mask = athletes['EVENT'].str.contains(r'100m', na=True)
athletes.loc[mask, 'MAPPED_EVENT'] = '100m'
mask = athletes['EVENT'].str.contains(r'200 Meter Dash', na=True)
athletes.loc[mask, 'MAPPED_EVENT'] = '200m'
mask = athletes['EVENT'].str.contains(r'200m', na=True)
athletes.loc[mask, 'MAPPED_EVENT'] = '200m'
mask = athletes['EVENT'].str.contains(r'300 Meter Run', na=True)
athletes.loc[mask, 'MAPPED_EVENT'] = '300m'
mask = athletes['EVENT'].str.contains(r'400 Meter Dash', na=True)
athletes.loc[mask, 'MAPPED_EVENT'] = '400m'
mask = athletes['EVENT'].str.contains(r'400m', na=True)
athletes.loc[mask, 'MAPPED_EVENT'] = '400m'
mask = athletes['EVENT'].str.contains(r'600 Meter Run', na=True)
athletes.loc[mask, 'MAPPED_EVENT'] = '600m'
mask = athletes['EVENT'].str.contains(r'800 Meter Dash', na=True)
athletes.loc[mask, 'MAPPED_EVENT'] = '800m'
mask = athletes['EVENT'].str.contains(r'800 Meter Run', na=True)
athletes.loc[mask, 'MAPPED_EVENT'] = '800m'
mask = athletes['EVENT'].str.contains(r'800m', na=True)
athletes.loc[mask, 'MAPPED_EVENT'] = '800m'
mask = athletes['EVENT'].str.contains(r'1500 Meter Run', na=True)
athletes.loc[mask, 'MAPPED_EVENT'] = '1500m'
mask = athletes['EVENT'].str.contains(r'1500m', na=True)
athletes.loc[mask, 'MAPPED_EVENT'] = '1500m'
mask = athletes['EVENT'].str.contains(r'3000 Meter Run', na=True)
athletes.loc[mask, 'MAPPED_EVENT'] = '3000m'
mask = athletes['EVENT'].str.contains(r'3000m', na=True)
athletes.loc[mask, 'MAPPED_EVENT'] = '3000m'
mask = athletes['EVENT'].str.contains(r'5000 Meter Run', na=True)
athletes.loc[mask, 'MAPPED_EVENT'] = '5000m'
mask = athletes['EVENT'].str.contains(r'5000m', na=True)
athletes.loc[mask, 'MAPPED_EVENT'] = '5000m'
mask = athletes['EVENT'].str.contains(r'10000 Meter Run', na=True)
athletes.loc[mask, 'MAPPED_EVENT'] = '10000m'
mask = athletes['EVENT'].str.contains(r'10000m', na=True)
athletes.loc[mask, 'MAPPED_EVENT'] = '10000m'
mask = athletes['EVENT'].str.contains(r'1 Mile Run', na=True)
athletes.loc[mask, 'MAPPED_EVENT'] = '1 mile'


# Correct hurdles

mask = athletes['EVENT'].str.contains(r'80m Hurdles', na=True)
athletes.loc[mask, 'MAPPED_EVENT'] = '80m hurdles'
mask = athletes['EVENT'].str.contains(r'80m hurdles', na=True)
athletes.loc[mask, 'MAPPED_EVENT'] = '80m hurdles'
mask = athletes['EVENT'].str.contains(r'80 Meter Hurdles', na=True)
athletes.loc[mask, 'MAPPED_EVENT'] = '80m hurdles'
mask = athletes['EVENT'].str.contains(r'100m Hurdles', na=True)
athletes.loc[mask, 'MAPPED_EVENT'] = '100m hurdles'
mask = athletes['EVENT'].str.contains(r'100 Meter Hurdles', na=True)
athletes.loc[mask, 'MAPPED_EVENT'] = '100m hurdles'
mask = athletes['EVENT'].str.contains(r'110m Hurdles', na=True)
athletes.loc[mask, 'MAPPED_EVENT'] = '110m hurdles'
mask = athletes['EVENT'].str.contains(r'110 Meter Hurdles', na=True)
athletes.loc[mask, 'MAPPED_EVENT'] = '110m hurdles'
mask = athletes['EVENT'].str.contains(r'200m Hurdles', na=True)
athletes.loc[mask, 'MAPPED_EVENT'] = '200m hurdles'
mask = athletes['EVENT'].str.contains(r'200 Meter Hurdles', na=True)
athletes.loc[mask, 'MAPPED_EVENT'] = '200m hurdles'
mask = athletes['EVENT'].str.contains(r'400m Hurdles', na=True)
athletes.loc[mask, 'MAPPED_EVENT'] = '400m hurdles'
mask = athletes['EVENT'].str.contains(r'400 Meter Hurdles', na=True)
athletes.loc[mask, 'MAPPED_EVENT'] = '400m hurdles'


# Throws

mask = athletes['EVENT'].str.contains(r'Javelin', na=True)
athletes.loc[mask, 'MAPPED_EVENT'] = 'Javelin throw'
mask = athletes['EVENT'].str.contains(r'Shot', na=True)
athletes.loc[mask, 'MAPPED_EVENT'] = 'Shot put'
mask = athletes['EVENT'].str.contains(r'Hammer', na=True)
athletes.loc[mask, 'MAPPED_EVENT'] = 'Hammer throw'
mask = athletes['EVENT'].str.contains(r'Discus', na=True)
athletes.loc[mask, 'MAPPED_EVENT'] = 'Discus throw'

# Jumps

mask = athletes['EVENT'].str.contains(r'High Jump', na=True)
athletes.loc[mask, 'MAPPED_EVENT'] = 'High jump'
mask = athletes['EVENT'].str.contains(r'Long Jump', na=True)
athletes.loc[mask, 'MAPPED_EVENT'] = 'Long jump'
mask = athletes['EVENT'].str.contains(r'Triple Jump', na=True)
athletes.loc[mask, 'MAPPED_EVENT'] = 'Triple jump'
mask = athletes['EVENT'].str.contains(r'Pole Vault', na=True)
athletes.loc[mask, 'MAPPED_EVENT'] = 'Pole vault'
mask = athletes['EVENT'].str.contains(r'High jump', na=True)
athletes.loc[mask, 'MAPPED_EVENT'] = 'High jump'
mask = athletes['EVENT'].str.contains(r'Long jump', na=True)
athletes.loc[mask, 'MAPPED_EVENT'] = 'Long jump'
mask = athletes['EVENT'].str.contains(r'Triple jump', na=True)
athletes.loc[mask, 'MAPPED_EVENT'] = 'Triple jump'
mask = athletes['EVENT'].str.contains(r'Pole vault', na=True)
athletes.loc[mask, 'MAPPED_EVENT'] = 'Pole vault'

# Steeplechase

mask = athletes['EVENT'].str.contains(r'2000m S/C', na=True)
athletes.loc[mask, 'MAPPED_EVENT'] = '2000m steeplechase'
mask = athletes['EVENT'].str.contains(r'2000m steeplechase', na=True)
athletes.loc[mask, 'MAPPED_EVENT'] = '2000m steeplechase'
mask = athletes['EVENT'].str.contains(r'2000 Meter Steeplechase', na=True)
athletes.loc[mask, 'MAPPED_EVENT'] = '2000m steeplechase'
mask = athletes['EVENT'].str.contains(r'3000m S/C', na=True)
athletes.loc[mask, 'MAPPED_EVENT'] = '3000m steeplechase'
mask = athletes['EVENT'].str.contains(r'3000 Meter Steeplechase', na=True)
athletes.loc[mask, 'MAPPED_EVENT'] = '3000m steeplechase'


# Walk

mask = athletes['EVENT'].str.contains(r'1500 Meter Race Walk', na=True)
athletes.loc[mask, 'MAPPED_EVENT'] = '1500m race walk'
mask = athletes['EVENT'].str.contains(r'3000m Race Walk', na=True)
athletes.loc[mask, 'MAPPED_EVENT'] = '3000m race walk'
mask = athletes['EVENT'].str.contains(r'3000 Meter Race Walk', na=True)
athletes.loc[mask, 'MAPPED_EVENT'] = '3000m race walk'
mask = athletes['EVENT'].str.contains(r'5000 Meter Race Walk', na=True)
athletes.loc[mask, 'MAPPED_EVENT'] = '5000m race walk'
mask = athletes['EVENT'].str.contains(r'5000m Race Walk', na=True)
athletes.loc[mask, 'MAPPED_EVENT'] = '5000m race walk'
mask = athletes['EVENT'].str.contains(r'10000 Meter Race Walk', na=True)
athletes.loc[mask, 'MAPPED_EVENT'] = '10000m race walk'

# Relay

mask = athletes['EVENT'].str.contains(r'4x100m Relay', na=True)
athletes.loc[mask, 'MAPPED_EVENT'] = '4 x 100m relay'
mask = athletes['EVENT'].str.contains(r'4 X 100m Relay', na=True)
athletes.loc[mask, 'MAPPED_EVENT'] = '4 x 100m relay'
mask = athletes['EVENT'].str.contains(r'4x400m Relay', na=True)
athletes.loc[mask, 'MAPPED_EVENT'] = '4 x 400m relay'
mask = athletes['EVENT'].str.contains(r'4 X 400m Relay', na=True)
athletes.loc[mask, 'MAPPED_EVENT'] = '4 x 400m relay'
mask = athletes['EVENT'].str.contains(r'4x100 Meter Relay', na=True)
athletes.loc[mask, 'MAPPED_EVENT'] = '4 x 100m relay'
mask = athletes['EVENT'].str.contains(r'4x400 Meter Relay', na=True)
athletes.loc[mask, 'MAPPED_EVENT'] = '4 x 400m relay'



### PROCESS BENCHMARKS ###

benchmarks = client.query_and_wait(benchmark_sql).to_dataframe()

benchmarks.rename(columns = {'RESULT':'BENCHMARK'}, inplace = True)
benchmarks.drop(['NAME', 'RANK', 'CATEGORY_EVENT', 'COMPETITION', 'STAGE'], axis=1, inplace=True)

# convert times in benchmarks to standard format

for i in range(len(benchmarks)):
        
    rowIndex = benchmarks.index[i]

    input_string=benchmarks.iloc[rowIndex,1]
    
    metric=benchmarks.iloc[rowIndex,0]
    
    if metric==None:
        continue
        
    out = convert_time(i, input_string, metric)
     
    benchmarks.loc[rowIndex, 'Metric'] = out

# Calculate benchmarks

mask = benchmarks['EVENT'].str.contains(r'jump|throw|Pole|put', na=True)

benchmarks.loc[mask, '2pc']=benchmarks['Metric']*0.98
benchmarks.loc[mask, '35pc']=benchmarks['Metric']*0.965
benchmarks.loc[mask, '5pc']=benchmarks['Metric']*0.95

benchmarks.loc[~mask, '2pc']=benchmarks['Metric']*1.02
benchmarks.loc[~mask, '35pc']=benchmarks['Metric']*1.035
benchmarks.loc[~mask, '5pc']=benchmarks['Metric']*1.05

benchmarks['MAPPED_EVENT']=benchmarks['EVENT']

# Merge benchmarks with df

df = athletes.reset_index().merge(benchmarks.reset_index(), on=['MAPPED_EVENT','GENDER'], how='left')
df['RESULT'] = athletes['RESULT'].replace(regex=r'–', value=np.nan)

# Convert df results into seconds format

for i in range(len(df)):
    
        
    rowIndex = df.index[i]

    input_string=df.iloc[rowIndex,5]    
    
    metric=df.iloc[rowIndex,2]
    
    if metric=='—' or metric=='DQ' or metric=='SCR' or metric=='FS' or metric=='DNQ' or metric==' DNS' or metric=='NH':
        continue
    
        
    out = convert_time(i, input_string, metric)
         
    df.loc[rowIndex, 'RESULT_CONV'] = out

df["AGE"].fillna(0, inplace=True)
df['AGE'] = df['AGE'].astype('float')

# Apply OCTC criteria

rslt_df = df.loc[(((df['CATEGORY_EVENT']=='Mid')|(df['CATEGORY_EVENT']=='Sprint')|(df['CATEGORY_EVENT']=='Long')|(df['CATEGORY_EVENT']=='Hurdles')|(df['CATEGORY_EVENT']=='Walk')|(df['CATEGORY_EVENT']=='Relay')|(df['CATEGORY_EVENT']=='Marathon')|(df['CATEGORY_EVENT']=='Steeple')|(df['CATEGORY_EVENT']=='Pentathlon')|(df['CATEGORY_EVENT']=='Heptathlon')|(df['CATEGORY_EVENT']=='Triathlon'))&(df['RESULT_CONV'] <= df['5pc']) & (df['AGE']<40) & ((df['MAPPED_EVENT']!='Marathon')|(df['AGE']<60) & (df['MAPPED_EVENT']=='Marathon')))|(((df['CATEGORY_EVENT']=='Jump')|(df['CATEGORY_EVENT']=='Throw'))&(df['RESULT_CONV'] >= df['5pc']) & (df['AGE']<40) & ((df['MAPPED_EVENT']!='Marathon')|(df['AGE']<60) & (df['MAPPED_EVENT']=='Marathon')))]

# Measure against 2%, 3.5% and 5% of SEAG 3rd place

mask = rslt_df['CATEGORY_EVENT'].str.contains(r'Jump|Throw', na=True)
rslt_df.loc[mask, 'Delta2'] = rslt_df['RESULT_CONV']-rslt_df['2pc']
rslt_df.loc[mask, 'Delta35'] = rslt_df['RESULT_CONV']-rslt_df['35pc']
rslt_df.loc[mask, 'Delta5'] = rslt_df['RESULT_CONV']-rslt_df['5pc']

rslt_df.loc[~mask, 'Delta2'] =  rslt_df['2pc'] - rslt_df['RESULT_CONV']
rslt_df.loc[~mask, 'Delta35'] = rslt_df['35pc'] - rslt_df['RESULT_CONV']
rslt_df.loc[~mask, 'Delta5'] = rslt_df['5pc'] - rslt_df['RESULT_CONV']

rslt_df=rslt_df.loc[rslt_df['COMPETITION']!='SEA Games']

# Name corrections

rslt_df['NAME'] = rslt_df['NAME'].replace(regex=r'PRAHARSH, RYAN', value='S/O SUBASH SOMAN, PRAHARSH RYAN')
rslt_df['NAME'] = rslt_df['NAME'].replace(regex=r'Ryan, Praharsh', value='S/O SUBASH SOMAN, PRAHARSH RYAN')
rslt_df['NAME'] = rslt_df['NAME'].replace(regex=r'TAN, ELIZABETH ANN SHEE R', value='TAN, ELIZABETH-ANN')
rslt_df['NAME'] = rslt_df['NAME'].replace(regex=r'Tan, Elizabeth Ann', value='TAN, ELIZABETH-ANN')

rslt_df['NAME'] = rslt_df['NAME'].replace(regex=r'LOUIS, MARC BRIAN', value='Louis, Marc Brian')
rslt_df['NAME'] = rslt_df['NAME'].replace(regex=r'Louis, Marc', value='Louis, Marc Brian')
rslt_df['NAME'] = rslt_df['NAME'].replace(regex=r'TAN JUN JIE', value='Tan Jun Jie')

rslt_df['NAME'] = rslt_df['NAME'].replace(regex=r'SNG, MICHELLE', value='Sng, Michelle')
rslt_df['NAME'] = rslt_df['NAME'].replace(regex=r'SNG, SUAT LI, MICHELLE', value='Sng, Michelle')

rslt_df['NAME'] = rslt_df['NAME'].replace(regex=r'MUN, IVAN', value='Mun, Ivan')
rslt_df['NAME'] = rslt_df['NAME'].replace(regex=r'LOW, JUN YU', value='Low, Jun Yu')

rslt_df['NAME'] = rslt_df['NAME'].replace(regex=r'ANG, CHEN XIANG', value='Ang, Chen Xiang')
rslt_df['NAME'] = rslt_df['NAME'].replace(regex=r'LIM, OLIVER', value='Lim, Oliver')

# Create scalar to measure relative performance

rslt_df['PERF_SCALAR']=rslt_df['Delta5']/rslt_df['Metric']*100

# Define SPEX carded athletes

spex_athletes_casefold = ['goh chui ling',
 'michelle sng',
 'quek jun jie calvin',
 'soh rui yong, guillaume',
 'aaron justin tan wen jie',
 'daniel leow soon yee',
 'joshua chua',
 'ng zhi rong ryan raphael',
 'wenli rachel',
 'wong yaohan melvin',
 'xander ho ann heng',
 'veronica shanti pereira',
 'ang chen xiang',
 'kampton kam',
 'marc brian louis',
 'mark lee ren',
 'reuben rainer lee siong en',
 'elizabeth-ann tan shee ru',
 'thiruben thana rajan',
 'bhavna gopikrishna',
 'chloe chee en-ya',
 'conrad kangli emery',
 'harry irfan curran',
 'huang weijun',
 'jayden tan',
 'koh shun yi audrey',
 'laavinia d/o jaiganth',
 'lim yee chern clara',
 'loh ding rong anson',
 'ong ying tat',
 'song en xu reagan',
 'subaraghav hari',
 'teh ying shan',
 'yan teo',
 'zhong chuhan',
 'esther tay shee wei',
 'faith ford',
 'garrett chua je-an',
 'lucas fun',
 'goh, chui ling',
 'sng, michelle',
 'quek, jun jie calvin',
 'soh rui yong, guillaume',
 'tan wen jie, aaron justin',
 'yee, daniel leow soon',
 'chua, joshua',
 'ng zhi rong, ryan raphael',
 'wenli, rachel',
 'wong yaohan, melvin',
 'ho ann heng,  xander',
 'pereira, veronica shanti',
 'ang, chen xiang',
 'kam, kampton',
 'marc brian louis',
 'mark lee ren',
 'lee siong en, reuben rainer',
 ' tan shee ru, elizabeth-ann',
 'thiruben thana rajan',
 'bhavna gopikrishna',
 'chee en-ya, chloe',
 'conrad kangli emery',
 'harry irfan curran',
 'huang, weijun',
 'tan, jayden',
 'koh shun yi, audrey',
 'laavinia d/o jaiganth',
 'lim yee chern, clara',
 'loh ding rong, anson',
 'ong, ying tat',
 'song en xu, reagan',
 'subaraghav hari',
 'teh, ying shan',
 'teo, yan',
 'zhong, chuhan',
 'tay shee wei, esther',
 'ford, faith',
 'chua je-an, garrett',
 'fun, lucas',
 'raphael, ryan',
 'ho, xander, ann heng',
 'louis, marc brian',
 'lee, mark ren',
 'lee, reuben rainer',
 'tan, elizabeth-ann',
 'irfan curran, harry',
 'huang, wei jun',
 'clara lim yee chern',
 'song, reagan en xu',
 'zhong chu han',
 'louis, marc',
 'tan, elizabeth ann shee r',
 'huang wei jun',
 'zhong, chu han']

top_performers=rslt_df.sort_values(['NAME','PERF_SCALAR'],ascending=False).groupby('NAME').head(1) # Choose top performing event per NAME

spexed_list = top_performers.loc[~rslt_df['NAME'].str.casefold().isin(spex_athletes_casefold)]  # ~ means NOT IN. DROP spex carded athletes

spexed_list.sort_values(['MAPPED_EVENT', 'GENDER', 'PERF_SCALAR'], ascending=[True, True, False], inplace=True)
spexed_list['overall_rank'] = 1
spexed_list['overall_rank'] = spexed_list.groupby(['MAPPED_EVENT', 'GENDER'])['overall_rank'].cumsum()

#Apply OCTC selection rule: max 6 for 100m/400m and max 3 for all other events

spexed_list=spexed_list[(((spexed_list['MAPPED_EVENT']=='400m')|(spexed_list['MAPPED_EVENT']=='100m'))&(spexed_list['overall_rank']<7))|(~((spexed_list['MAPPED_EVENT']=='400m')|(spexed_list['MAPPED_EVENT']=='100m'))&(spexed_list['overall_rank']<4))]


# Show resulting dataframe

st.write(spexed_list)



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
