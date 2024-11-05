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

#rows = run_query("SELECT word FROM `bigquery-public-data.samples.shakespeare` LIMIT 10")

#rows = run_query("SELECT * FROM `saa-analytics.results.saa_full` LIMIT 10")


# Print results.
#st.write("Writing rows from table:")
#for row in rows:
#    st.write("✍️ " + row['word'])
#    st.write(row)

benchmark_sql = """
SELECT NAME, RESULT, RANK, EVENT, CATEGORY_EVENT, GENDER, COMPETITION, STAGE
FROM `saa-analytics.results.saa_full`
WHERE STAGE='Final' AND COMPETITION='SEA Games' AND RANK='3'
"""

df = client.query_and_wait("""SELECT * FROM `saa-analytics.results.saa_full`""").to_dataframe()

df.dropna(how= "all", axis=1, inplace=True)

year_list = df['DATE'].unique().tolist() # get unique list of source events
region_list = df['REGION'].unique().tolist()
competition_list = df['COMPETITION'].unique().tolist()

year_selection = st.multiselect(
    "Please select the desired year(s):",
    year_list,
)

region_selection = st.multiselect(
    "Please select the desired region(S):",
    region_list,
)

competition_selection = st.multiselect(
    "Please select the desired competition(s):",
    competition_list,
)


df_filtered = df[df['DATE'].isin(year_selection) & df['REGION'].isin(region_selection) & df['COMPETITION'].isin(competition_selection)]

st.write(df_filtered)

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
df['RESULT'] = athletes['RESULT'].replace(regex=r'–', value=np.NaN)

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

st.write(benchmarks)


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
