# streamlit_app.py

import pandas as pd
import streamlit as st
import datetime
import numpy as np
import re



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

            elif string=='':   # no event description at all!
                
                output='' # return nothing
            
        
            else:
        
                searchstring = ":"
                searchstring2 = "."
                substring=str(metric)
                count = substring.count(searchstring)
                count2 = substring.count(searchstring2)
            
                if count==0:
                
                    output=float(substring)

                elif '10,000m' in string and count==2:  # fix erroneous timing format from XX:XX:XX to XX:XX.XX
                
                
                    idx = 5 # 6th character position
                    replacement = "."
                    metric = metric[:idx] + replacement + metric[idx+1:]                
                
                    m,s = metric.split(':')            

                    output = float(datetime.timedelta(minutes=int(m),seconds=float(s)).total_seconds())
                    

                elif '1500m' in string and count==2:  # fix erroneous timing format from XX:XX:XX to XX:XX.XX
                    
                    if len(substring)==7:  # format is X:XX:XX and not XX:XX:XX 
                        
                        idx = 4 # 5th character position
                        replacement = "."
                        metric = '0' + metric[:idx] + replacement + metric[idx+1:]                
                
                        m,s = metric.split(':')            

                        output = float(datetime.timedelta(minutes=int(m),seconds=float(s)).total_seconds())
                    
                        
                    else:  # format is XX:XX:XX
                        
                        idx = 5 # 5th character position
                        replacement = "."
                        metric = metric[:idx] + replacement + metric[idx+1:]                
                
                        m,s = metric.split(':')            

                        output = float(datetime.timedelta(minutes=int(m),seconds=float(s)).total_seconds())
                    
                    
        
                       
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

        input_string=df.loc[rowIndex,'EVENT']
    
        metric=df.loc[rowIndex,'BENCHMARKS']
    
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

        input_string=df.loc[rowIndex,'MAPPED_EVENT']    # event description
    
        metric=df.iloc[rowIndex,'RESULT'] # result
    
        if metric=='—' or metric=='DQ' or metric=='SCR' or metric=='FS' or metric=='DNQ' or metric=='DNS' or metric=='NH' or metric=='NM' or metric=='FOUL' or metric=='DNF' or metric=='SR' :
            continue
    
        result_out = convert_time(i, input_string, metric)
 #   print('line', i, input_string, metric, result_out)
         
        df.loc[rowIndex, 'RESULT_CONV'] = result_out

    return df

@st.cache_data
def map_events(athletes):

    # Create temporary mapped event column

   # athletes['MAPPED_EVENT']=''

## Clear columns of special characters and spaces

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

    return athletes

@st.cache_data
def event_date(df):

    for i in range(len(df)):
        
        rowIndex = df.index[i]

        date = df.loc[rowIndex,'DATE']
        year = df.loc[rowIndex,'YEAR']    
    
        if 'to' in date or ' - ' in date:
        
            pos = re.search('to|\s\-\s', date)
                                                
        # Splice string to day and month

            split_pos_start=pos.start()+3

            
            final_date = date[split_pos_start:] # left string post splicing
            final_year = year[2:]
        
            event_date = final_date + '/' + final_year
        
            df.loc[rowIndex, 'event_date'] = event_date
        
        elif re.search('\w\-\w', date):
        
            if df.iloc[rowIndex, 15] == "National School Games":
        
                event_date = '04'+'/'+date[1:3] + '/' + year[2:]  # reverse order from dd/mm to mm/dd
        
                df.loc[rowIndex, 'event_date'] = event_date
        
            else:
            
                event_date = date + '-' + year[2:]
            
                df.loc[rowIndex, 'event_date'] = event_date

        
        

        
    df['event_date'] = df['event_date'].astype(str)
    df['event_date'] = df['event_date'].str.replace('\xa0', ' ', regex=True)
    df['event_date'] = df['event_date'].str.replace('[\x00-\x1f\x7f-\x9f]', '', regex=True)
    df['event_date'] = df['event_date'].str.replace('\r', ' ', regex=True)
    df['event_date'] = df['event_date'].str.replace('\n', ' ', regex=True)
    df['event_date'] = df['event_date'].str.strip()

    return df
       
    

