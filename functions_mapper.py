import pandas as pd
import streamlit as st
import datetime
import numpy as np
import re
import gcsfs
from st_files_connection import FilesConnection


WITH cleaned_athletes AS (
  SELECT
    -- Clean all relevant string columns similarly; example shown for main ones
    TRIM(REGEXP_REPLACE(REGEXP_REPLACE(REGEXP_REPLACE(REGEXP_REPLACE(CAST(EVENT AS STRING), r'\xA0', ' '), r'[\x00-\x1F\x7F-\x9F]', ''), r'\r', ' '), r'\n', ' ')) AS EVENT,
    TRIM(REGEXP_REPLACE(REGEXP_REPLACE(REGEXP_REPLACE(REGEXP_REPLACE(CAST(DISTANCE AS STRING), r'\xA0', ' '), r'[\x00-\x1F\x7F-\x9F]', ''), r'\r', ' '), r'\n', ' ')) AS DISTANCE,
    TRIM(REGEXP_REPLACE(REGEXP_REPLACE(REGEXP_REPLACE(REGEXP_REPLACE(CAST(EVENT_CLASS AS STRING), r'\xA0', ' '), r'[\x00-\x1F\x7F-\x9F]', ''), r'\r', ' '), r'\n', ' ')) AS EVENT_CLASS,
    TRIM(REGEXP_REPLACE(REGEXP_REPLACE(REGEXP_REPLACE(REGEXP_REPLACE(CAST(GENDER AS STRING), r'\xA0', ' '), r'[\x00-\x1F\x7F-\x9F]', ''), r'\r', ' '), r'\n', ' ')) AS GENDER,
    TRIM(REGEXP_REPLACE(REGEXP_REPLACE(REGEXP_REPLACE(REGEXP_REPLACE(CAST(DIVISION AS STRING), r'\xA0', ' '), r'[\x00-\x1F\x7F-\x9F]', ''), r'\r', ' '), r'\n', ' ')) AS DIVISION,
    TRIM(REGEXP_REPLACE(REGEXP_REPLACE(REGEXP_REPLACE(REGEXP_REPLACE(CAST(REGION AS STRING), r'\xA0', ' '), r'[\x00-\x1F\x7F-\x9F]', ''), r'\r', ' '), r'\n', ' ')) AS REGION,
    *
  FROM athletes
),

mapped_athletes AS (
  SELECT
    *,
    -- Correct Javelin category
    CASE 
      WHEN REGEXP_CONTAINS(EVENT, r'Javelin') THEN 'Throw'
      ELSE CATEGORY_EVENT
    END AS CATEGORY_EVENT_CORRECTED,

    -- Map Running Events to standardized MAPPED_EVENT
    CASE
      WHEN (REGEXP_CONTAINS(EVENT, r'Dash') AND REGEXP_CONTAINS(DISTANCE, r'100')) THEN '100m'
      WHEN (REGEXP_CONTAINS(EVENT, r'Run') AND REGEXP_CONTAINS(DISTANCE, r'100')) THEN '100m'
      WHEN REGEXP_CONTAINS(EVENT, r'100 Meter Run') THEN '100m'
      WHEN EVENT = '100m' THEN '100m'

      WHEN (REGEXP_CONTAINS(EVENT, r'Dash') AND REGEXP_CONTAINS(DISTANCE, r'200')) THEN '200m'
      WHEN (REGEXP_CONTAINS(EVENT, r'Run') AND REGEXP_CONTAINS(DISTANCE, r'200')) THEN '200m'
      WHEN REGEXP_CONTAINS(EVENT, r'^200m$') THEN '200m'
      WHEN REGEXP_CONTAINS(EVENT, r'200\sMeter') THEN '200m'

      WHEN (REGEXP_CONTAINS(EVENT, r'Dash') AND REGEXP_CONTAINS(DISTANCE, r'400')) THEN '400m'
      WHEN REGEXP_CONTAINS(EVENT, r'^400m$') THEN '400m'
      WHEN REGEXP_CONTAINS(EVENT, r'^400\sMeter$') THEN '400m'
      WHEN (REGEXP_CONTAINS(EVENT, r'Run') AND REGEXP_CONTAINS(DISTANCE, r'400')) THEN '400m'

      WHEN (REGEXP_CONTAINS(EVENT, r'Run') AND REGEXP_CONTAINS(DISTANCE, r'800')) THEN '800m'
      WHEN REGEXP_CONTAINS(EVENT, r'800 Meter Run') THEN '800m'
      WHEN REGEXP_CONTAINS(EVENT, r'^800m$') THEN '800m'
      WHEN (REGEXP_CONTAINS(EVENT, r'Run') AND REGEXP_CONTAINS(DISTANCE, r'1000')) THEN '1000m'

      WHEN (REGEXP_CONTAINS(EVENT, r'Run') AND REGEXP_CONTAINS(DISTANCE, r'1500')) THEN '1500m'
      WHEN REGEXP_CONTAINS(EVENT, r'^1500m$') THEN '1500m'

      WHEN (REGEXP_CONTAINS(EVENT, r'Run') AND REGEXP_CONTAINS(DISTANCE, r'3000')) THEN '3000m'
      WHEN (REGEXP_CONTAINS(EVENT, r'Run') AND REGEXP_CONTAINS(DISTANCE, r'5000')) THEN '5000m'
      WHEN REGEXP_CONTAINS(EVENT, r'^5000m$') THEN '5000m'

      WHEN (REGEXP_CONTAINS(EVENT, r'Run') AND REGEXP_CONTAINS(DISTANCE, r'10000')) THEN '10,000m'
      WHEN REGEXP_CONTAINS(EVENT, r'^10000m$') THEN '10,000m'
      WHEN REGEXP_CONTAINS(EVENT, r'^10\,000m$') THEN '10,000m'

      WHEN (REGEXP_CONTAINS(EVENT, r'Run') AND REGEXP_CONTAINS(DISTANCE, r'Mile')) THEN '1 Mile'

      -- Hurdles examples
      WHEN (REGEXP_CONTAINS(EVENT, r'100m Hurdles|100m hurdles') AND REGEXP_CONTAINS(EVENT_CLASS, r'0.84') AND REGEXP_CONTAINS(GENDER, r'Female')) THEN '100m Hurdles'
      WHEN (REGEXP_CONTAINS(EVENT, r'^Hurdles$') AND REGEXP_CONTAINS(DISTANCE, r'110') AND REGEXP_CONTAINS(DIVISION, r'OPEN|Open') AND REGEXP_CONTAINS(GENDER, r'Male')) THEN '110m Hurdles'
      WHEN (REGEXP_CONTAINS(EVENT, r'^Hurdles$') AND REGEXP_CONTAINS(DISTANCE, r'400') AND REGEXP_CONTAINS(EVENT_CLASS, r'0.914|0.762') AND REGEXP_CONTAINS(GENDER, r'Female|Male')) THEN '400m Hurdles'

      -- Throws examples
      WHEN (REGEXP_CONTAINS(EVENT, r'Javelin') AND REGEXP_CONTAINS(EVENT_CLASS, r'600g|800g') AND REGEXP_CONTAINS(GENDER, r'Female|Male')) THEN 'Javelin Throw'
      WHEN (REGEXP_CONTAINS(EVENT, r'Shot Put') AND ((GENDER = 'Female' AND REGEXP_CONTAINS(EVENT_CLASS, r'4kg')) OR (GENDER = 'Male' AND REGEXP_CONTAINS(EVENT_CLASS, r'7.26kg')))) THEN 'Shot Put'
      WHEN (REGEXP_CONTAINS(EVENT, r'Hammer Throw') AND REGEXP_CONTAINS(EVENT_CLASS, r'7.26kg|4.00kg')) THEN 'Hammer Throw'
      WHEN (REGEXP_CONTAINS(EVENT, r'Discus Throw|Discus') AND ((GENDER = 'Male' AND REGEXP_CONTAINS(EVENT_CLASS, r'2kg')) OR (GENDER = 'Female' AND REGEXP_CONTAINS(EVENT_CLASS, r'1kg')))) THEN 'Discus Throw'

      -- Jumps
      WHEN REGEXP_CONTAINS(EVENT, r'High Jump') THEN 'High Jump'
      WHEN REGEXP_CONTAINS(EVENT, r'^Long\sJump$') THEN 'Long Jump'
      WHEN REGEXP_CONTAINS(EVENT, r'Triple Jump') THEN 'Triple Jump'
      WHEN REGEXP_CONTAINS(EVENT, r'Pole Vault') THEN 'Pole Vault'

      -- Steeplechase
      WHEN (REGEXP_CONTAINS(EVENT, r'3000m Steeplechase|3000m S\/C') AND REGEXP_CONTAINS(REGION, r'International')) THEN '3000m Steeplechase'

      -- Marathon
      WHEN REGEXP_CONTAINS(EVENT, r'^Marathon$') THEN 'Marathon'
      WHEN REGEXP_CONTAINS(EVENT, r'^Half\sMarathon$') THEN 'Half Marathon'

      -- Racewalk
      WHEN (REGEXP_CONTAINS(EVENT, r'Race Walk') AND REGEXP_CONTAINS(DISTANCE, r'10000')) THEN '10000m Racewalk'

      -- Relay
      WHEN REGEXP_CONTAINS(EVENT, r'4x80m Relay|4 x 80m') THEN '4 x 80m'
      WHEN REGEXP_CONTAINS(EVENT, r'4x100m Relay|4 x 100m|4 X 100m') THEN '4 x 100m'
      WHEN (REGEXP_CONTAINS(EVENT, r'Relay') AND REGEXP_CONTAINS(DISTANCE, r'400|1600')) THEN '4 x 400m'

      -- Decathlon/Heptathlon
      WHEN REGEXP_CONTAINS(EVENT, r'^Heptathlon$') THEN 'Heptathlon'
      WHEN REGEXP_CONTAINS(EVENT, r'^Decathlon$') THEN 'Decathlon'

      ELSE MAPPED_EVENT
    END AS MAPPED_EVENT_CORRECTED
  FROM cleaned_athletes;


  '''
  SELECT
  *,
  CASE
    -- Skip marks with illegal wind speeds (if 'w' in metric)
    WHEN LOWER(CAST(metric AS STRING)) LIKE '%w%' THEN NULL

    -- Field events (distance-based marks, remove 'm' or 'GR')
    WHEN LOWER(CAST(event AS STRING)) LIKE '%discus%' OR LOWER(CAST(event AS STRING)) LIKE '%throw%' OR
         LOWER(CAST(event AS STRING)) LIKE '%jump%' OR LOWER(CAST(event AS STRING)) LIKE '%vault%' OR
         LOWER(CAST(event AS STRING)) LIKE '%shot%' THEN 
      SAFE_CAST(REGEXP_REPLACE(REGEXP_REPLACE(CAST(metric AS STRING), r'(m|GR)', ''), r'\s', '') AS FLOAT64)

    -- No event description
    WHEN TRIM(CAST(event AS STRING)) = '' THEN NULL

    -- HH:MM:SS[.ss] format
    WHEN REGEXP_CONTAINS(CAST(metric AS STRING), r'^\d{1,2}:\d{2}:\d{2}(\.\d+)?$')
      THEN
          -- Convert to seconds: 3600*hour + 60*min + sec
          (SPLIT(metric, ':')[SAFE_OFFSET(0)]*3600) +
          (SPLIT(metric, ':')[SAFE_OFFSET(1)]*60) +
          SAFE_CAST(SPLIT(metric, ':')[SAFE_OFFSET(2)] AS FLOAT64)

    -- MM:SS[.ss] format
    WHEN REGEXP_CONTAINS(CAST(metric AS STRING), r'^\d{1,2}:\d{2}(\.\d+)?$')
      THEN
          -- Convert to seconds: 60*min + sec
          (SPLIT(metric, ':')[SAFE_OFFSET(0)]*60) +
          SAFE_CAST(SPLIT(metric, ':')[SAFE_OFFSET(1)] AS FLOAT64)

    -- Numeric (seconds, as float)
    WHEN REGEXP_CONTAINS(CAST(metric AS STRING), r'^\d+(\.\d+)?$')
      THEN SAFE_CAST(metric AS FLOAT64)

    ELSE NULL
  END AS parsed_metric

FROM athletes_table

'''
                                                      
