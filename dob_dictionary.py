import pandas as pd
import streamlit as st
import numpy as np
import gcsfs
import string

from st_files_connection import FilesConnection

def dob_dict(df):
  
  dob_df = df[df.DOB_dt.apply(lambda x: len(str(x))>2)] # filter out rows without DOB_dt 

  dictionary_dob = dict(zip(dob_df['NAME'].str.casefold(), dob_df['DOB_dt']))

  # Create dictionary version with punctuation stripped out

  translator = str.maketrans('', '', string.punctuation)
  dob_df['clean_name'] = dob_df['NAME'].str.translate(translator)
  dob_df['clean_name'] = dob_df['clean_name'].str.casefold()

  dictionary_dob_clean = dict(zip(dob_df['clean_name'].str.casefold(), dob_df['DOB_dt']))

  return dictionary_dob_clean



