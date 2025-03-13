# streamlit_app.py

import pandas as pd
import streamlit as st
import datetime


## Helper functions

# Convert results into standard format

@st.cache_data
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


## Convert time into seconds and distances into float

@st.cache_data
def process_benchmarks(df):
    
    for i in range(len(df)):

        rowIndex = df.index[i]

        input_string=df.iloc[rowIndex,0]
    
        metric=df.iloc[rowIndex,3]
    
        if metric==None:
        
            continue
        
        out = convert_time(i, input_string, metric)
        
        print(rowIndex, input_string, out)

    
        df.loc[rowIndex, 'Metric'] = out
    
    return df

@st.cache_data
def process_results(df):

    for i in range(len(df)):
    
        result_out=''
        
        rowIndex = df.index[i]

        input_string=df.iloc[rowIndex,14]    # event description
    
        metric=df.iloc[rowIndex,9] # result
    
        if metric=='—' or metric=='DQ' or metric=='SCR' or metric=='FS' or metric=='DNQ' or metric=='DNS' or metric=='NH' or metric=='NM' or metric=='FOUL' or metric=='DNF' or metric=='SR' :
            continue
    
        result_out = convert_time(i, input_string, metric)
 #   print('line', i, input_string, metric, result_out)
         
        df.loc[rowIndex, 'RESULT_CONV'] = result_out

    return df

