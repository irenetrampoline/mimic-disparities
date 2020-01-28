import numpy as np
import pandas as pd
import psycopg2
from scipy.stats import ks_2samp
import os 
import random

random.seed(22891)

# create a database connection
sqluser = 'iychen'
dbname = 'mimic'
schema_name = 'mimiciii'

# Connect to local postgres version of mimic
con = psycopg2.connect(dbname=dbname, user=sqluser)
cur = con.cursor()
cur.execute('SET search_path to ' + schema_name)

#========helper function for imputing missing values 

def replace(group):
    """
      takes in a pandas group, and replaces the 
      null value with the mean of the none null
      values of the same group 
    """
    mask = group.isnull()
    group[mask] = group[~mask].mean()
    return group


#========get the icu details 

# this query extracts the following:
#   Unique ids for the admission, patient and icu stay 
#   Patient gender 
#   admission & discharge times 
#   length of stay 
#   age 
#   ethnicity 
#   admission type 
#   in hospital death?
#   in icu death?
#   one year from admission death?
#   first hospital stay 
#   icu intime, icu outime 
#   los in icu 
#   first icu stay?

denquery = \
"""
-- This query extracts useful demographic/administrative information for patient ICU stays
--DROP MATERIALIZED VIEW IF EXISTS icustay_detail CASCADE;
--CREATE MATERIALIZED VIEW icustay_detail as

--ie is the icustays table 
--adm is the admissions table 
SELECT ie.subject_id, ie.hadm_id, ie.icustay_id
, pat.gender
, adm.admittime, adm.dischtime, adm.diagnosis
, ROUND( (CAST(adm.dischtime AS DATE) - CAST(adm.admittime AS DATE)) , 4) AS los_hospital
, ROUND( (CAST(adm.admittime AS DATE) - CAST(pat.dob AS DATE))  / 365, 4) AS age
, adm.ethnicity, adm.ADMISSION_TYPE
--, adm.hospital_expire_flag
, adm.insurance
, CASE when adm.deathtime between adm.admittime and adm.dischtime THEN 1 ELSE 0 END AS mort_hosp
, CASE when adm.deathtime between ie.intime and ie.outtime THEN 1 ELSE 0 END AS mort_icu
, CASE when adm.deathtime between adm.admittime and adm.admittime + interval '365' day  THEN 1 ELSE 0 END AS mort_oneyr
, DENSE_RANK() OVER (PARTITION BY adm.subject_id ORDER BY adm.admittime) AS hospstay_seq
, CASE
    WHEN DENSE_RANK() OVER (PARTITION BY adm.subject_id ORDER BY adm.admittime) = 1 THEN 1
    ELSE 0 END AS first_hosp_stay
-- icu level factors
, ie.intime, ie.outtime
, ie.FIRST_CAREUNIT
, ROUND( (CAST(ie.outtime AS DATE) - CAST(ie.intime AS DATE)) , 4) AS los_icu
, DENSE_RANK() OVER (PARTITION BY ie.hadm_id ORDER BY ie.intime) AS icustay_seq

-- first ICU stay *for the current hospitalization*
, CASE
    WHEN DENSE_RANK() OVER (PARTITION BY ie.hadm_id ORDER BY ie.intime) = 1 THEN 1
    ELSE 0 END AS first_icu_stay

FROM icustays ie
INNER JOIN admissions adm
    ON ie.hadm_id = adm.hadm_id
INNER JOIN patients pat
    ON ie.subject_id = pat.subject_id
WHERE adm.has_chartevents_data = 1
ORDER BY ie.subject_id, adm.admittime, ie.intime;

"""

den = pd.read_sql_query(denquery,con)

#----drop patients with less than 48 hour 
den['los_icu_hr'] = (den.outtime - den.intime).astype('timedelta64[h]')
den = den[(den.los_icu_hr >= 48)]
den = den[(den.age<300)]
den.drop('los_icu_hr', 1, inplace = True)

#----clean up

# micu --> medical 
# csru --> cardiac surgery recovery unit 
# sicu --> surgical icu 
# tsicu --> Trauma Surgical Intensive Care Unit
# NICU --> Neonatal 

den['adult_icu'] = np.where(den['first_careunit'].isin(['PICU', 'NICU']), 0, 1)
den['gender'] = np.where(den['gender']=="M", 1, 0)

# no need to yell 
den.ethnicity = den.ethnicity.str.lower()
den.ethnicity.loc[(den.ethnicity.str.contains('^white'))] = 'white'
den.ethnicity.loc[(den.ethnicity.str.contains('^black'))] = 'black'
den.ethnicity.loc[(den.ethnicity.str.contains('^hisp')) | (den.ethnicity.str.contains('^latin'))] = 'hispanic'
den.ethnicity.loc[(den.ethnicity.str.contains('^asia'))] = 'asian'
den.ethnicity.loc[~(den.ethnicity.str.contains('|'.join(['white', 'black', 'hispanic', 'asian'])))] = 'other'

den = pd.concat([den, pd.get_dummies(den['ethnicity'], prefix='eth')], 1)
den = pd.concat([den, pd.get_dummies(den['admission_type'], prefix='admType')], 1)

den.drop(['diagnosis', 'hospstay_seq', 'los_icu','icustay_seq', 'admittime', 'dischtime','los_hospital', 'intime', 'outtime', 'ethnicity', 'admission_type', 'first_careunit'], 1, inplace =True) 

den = den[(den['adult_icu']==1)].dropna()

notesquery = \
"""
SELECT fin.subject_id, fin.hadm_id, fin.icustay_id, string_agg(fin.text, ' ') as chartext
FROM (
  select ie.subject_id, ie.hadm_id, ie.icustay_id, ne.text
  from icustays ie
  left join noteevents ne
  on ie.subject_id = ne.subject_id and ie.hadm_id = ne.hadm_id 
  and ne.charttime between ie.intime and ie.intime + interval '48' hour
  --and ne.iserror != '1'
  where ne.category != 'Discharge summary'
) fin 

group by fin.subject_id, fin.hadm_id, fin.icustay_id 
order by fin.subject_id, fin.hadm_id, fin.icustay_id; 
"""

notes48 = pd.read_sql_query(notesquery,con) 

output_df = notes48.merge(den,on=['subject_id', 'hadm_id', 'icustay_id'], how='inner')

print('notes48:', len(notes48))
print('demographics:', len(den))
print('merged:', len(output_df))

output_df.to_csv('data/patients_notes.csv', index=False)

cols_except_notes = [i for i in output_df.columns if i != 'chartext']
output_df[cols_except_notes].to_csv('data/patients_info.csv', index=False)

print('saved to data/')