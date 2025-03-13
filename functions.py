# streamlit_app.py

import pandas as pd
import streamlit as st

## Helpers
@st.cache_data
def preprocess(i, string, metric):

    global OP

    l=['discus', 'throw', 'jump', 'vault', 'shot']

    string=string.lower()


    if any(s in string for s in l)==True:

        OP=float(str(metric))



    else:

        searchstring = ":"
        searchstring2 = "."
        substring=str(metric)
        count = substring.count(searchstring)
        count2 = substring.count(searchstring2)

        if count==0:
            OP=float(substring)



        elif (type(metric)==datetime.time or type(metric)==datetime.datetime):

            time=str(metric)
            h, m ,s = time.split(':')
            OP = float(datetime.timedelta(hours=int(h),minutes=int(m),seconds=float(s)).total_seconds())


        elif (count==1 and count2==1):

            m,s = metric.split(':')
            OP = float(datetime.timedelta(minutes=int(m),seconds=float(s)).total_seconds())

        elif (count==1 and count2==2):

            metric = metric.replace(".", ":", 1)

            h,m,s = metric.split(':')
            OP = float(datetime.timedelta(hours=int(h),minutes=int(m),seconds=float(s)).total_seconds())


        elif (count==2 and count2==0):

            h,m,s = metric.split(':')
            OP = float(datetime.timedelta(hours=int(h),minutes=int(m),seconds=float(s)).total_seconds())


    return OP


# Clean each row of input file
@st.cache_data
def clean(data):

    for i in range(len(data)):

        rowIndex = data.index[i]

        input_string=data.iloc[rowIndex,1]
        metric=data.iloc[rowIndex,5]

        processed_output = preprocess(i, input_string, metric)

        data.loc[rowIndex, 'Metric'] = processed_output

    return data
