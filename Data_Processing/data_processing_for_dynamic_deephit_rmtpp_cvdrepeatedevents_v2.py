# -*- coding: utf-8 -*-
"""
Created on Fri May  3 16:28:31 2019

@author: AdminCOOP

Reference: Automatic prediction of coronary artery disease from clinical narratives
         : https://github.com/MIT-LCP/mimic-code/blob/master/concepts/firstday/vitals-first-day.sql
"""

import pandas as pd
import numpy as np
import copy
import pickle
import sys
import os
from os import walk
import random
#import lime
#from lime import lime_tabular



np_rand_seed = 2

workdir = 'D:\\Work\\mimic_iii_deephit_rmtpp'
mimic_csv_folder = "mimic_iii_csv"
processed_folder = "processed"

# MIMIC-III filenames
patients_filename = os.path.join(workdir, mimic_csv_folder,"PATIENTS.csv")
admissions_filename = os.path.join(workdir, mimic_csv_folder,"ADMISSIONS.csv")
diagnoses_icd_filename = os.path.join(workdir, mimic_csv_folder,"DIAGNOSES_ICD.csv")
d_icd_diagnoses_filename = os.path.join(workdir, mimic_csv_folder,"D_ICD_DIAGNOSES.csv")
d_icd_procedures_filename = os.path.join(workdir, mimic_csv_folder,"D_ICD_PROCEDURES.csv")
procedures_icd_filename = os.path.join(workdir, mimic_csv_folder,"PROCEDURES_ICD.csv")
prescriptions_filename = os.path.join(workdir, mimic_csv_folder,"PRESCRIPTIONS.csv")

# Mapping Files
ndc2atc_file = os.path.join(workdir, mimic_csv_folder,"ndc2atc_level4.csv")
cid_atc = os.path.join(workdir, mimic_csv_folder,"drug-atc.csv")
ndc2rxnorm_file = os.path.join(workdir, mimic_csv_folder,"ndc2rxnorm_mapping.txt")

patients = pd.read_csv(patients_filename)
admissions = pd.read_csv(admissions_filename)
diagnoses_icd = pd.read_csv(diagnoses_icd_filename)
d_icd_diagnoses = pd.read_csv(d_icd_diagnoses_filename)
procedures_icd = pd.read_csv(procedures_icd_filename)
prescriptions = pd.read_csv(prescriptions_filename,dtype={'NDC':'category'})
      
# MACE Label
codedir = os.path.join(workdir, 'mace_icd9')
filelist = []
for (dirpath, dirnames, filenames) in walk(codedir):
    filelist.extend(filenames)
    break
filelist.sort()
for i in filelist:
    pickle_file = os.path.join(codedir,i)
    with open(pickle_file,'rb') as f:
        locals()[i[:-7]] = pickle.load(f)


label_file = copy.deepcopy(diagnoses_icd[['SUBJECT_ID','HADM_ID','ICD9_CODE']].drop_duplicates())
label_file['LABEL_MACE'] = 0
label_file['MACE_DEATH'] = 0
label_file['LABEL_CVD'] = 0
label_file.loc[label_file['ICD9_CODE'].isin(ami_mace_icd9 + stroke_mace_icd9_set1 + stroke_mace_icd9_set2 + stroke_mace_icd9_set3),'LABEL_MACE'] = 1
label_file.loc[label_file['ICD9_CODE'].isin(
mace_death_ami_icd9+mace_death_hf_icd9+
mace_death_pad_icd9+mace_death_pe_icd9+
mace_death_rupaa_icd9+
mace_death_stroke_icd9_set1+mace_death_stroke_icd9_set2+mace_death_stroke_icd9),'MACE_DEATH'] = 1
procedures_icd['ICD9_CODE'] = procedures_icd['ICD9_CODE'].astype(str)
hadm_cabg_pci = procedures_icd[procedures_icd['ICD9_CODE'].isin(outcome_cabg_procedure_codes + outcome_pci_procedure_codes)]['HADM_ID'].unique()
label_file.loc[label_file['HADM_ID'].isin(hadm_cabg_pci),'MACE_DEATH'] = 1
label_file.loc[label_file['ICD9_CODE'].isin(cvd_angina_icd9+cvd_angina_pectoris_icd9+cvd_hf_icd9+cvd_ischaemic_icd9+cvd_mi_icd9+cvd_pad_icd9+cvd_stroke_hae_icd9+cvd_stroke_isc_icd9+cvd_tia_icd9),'LABEL_CVD'] = 1

label_file['LABEL_DIABETES'] = 0
label_file.loc[label_file['ICD9_CODE'].isin(diabetes_icd9),'LABEL_DIABETES'] = 1
label_file = label_file.merge(admissions[['SUBJECT_ID','HADM_ID','ADMITTIME','DISCHTIME','DEATHTIME']].drop_duplicates(),on=['SUBJECT_ID','HADM_ID'],how='left')
label_file.loc[(label_file['MACE_DEATH']==1) & pd.notnull(label_file['DEATHTIME']),'LABEL_MACE'] = 1
label_file = label_file.drop(columns=['DEATHTIME'])
label_file = label_file.drop_duplicates()
label_file = label_file.merge(admissions[['SUBJECT_ID','DEATHTIME']].drop_duplicates(),on=['SUBJECT_ID'],how='left')
label_file['ADMITTIME'] = pd.to_datetime(label_file['ADMITTIME'])
label_file['DISCHTIME'] = pd.to_datetime(label_file['DISCHTIME'])
label_file['DEATHTIME'] = pd.to_datetime(label_file['DEATHTIME'])
label_file['DAYSDIFF'] = label_file['DEATHTIME'] - label_file['ADMITTIME']
label_file['DAYSDIFF'] = label_file['DAYSDIFF'].astype('timedelta64[D]')
label_file.loc[(label_file['MACE_DEATH']==1) & (label_file['DAYSDIFF'] <= 30) & (label_file['DAYSDIFF'] >= 0),'LABEL_MACE'] = 1
label_file = label_file.drop(columns=['ADMITTIME','DISCHTIME','DEATHTIME','DAYSDIFF'])
label_file = label_file.drop_duplicates()
label_file = label_file.drop(columns=['MACE_DEATH','ICD9_CODE'])
label_file = label_file.groupby(by=['SUBJECT_ID','HADM_ID']).max().reset_index()
label_file.loc[label_file['LABEL_MACE']==1,'LABEL_CVD'] = 0

diabetic_sub_id = label_file['SUBJECT_ID'].unique()#list(label_file[label_file['LABEL_DIABETES']==1]['SUBJECT_ID'].unique()) #

data_pat  = patients[patients['SUBJECT_ID'].isin(diabetic_sub_id)]
data_pat = data_pat[['SUBJECT_ID','GENDER','DOB']].drop_duplicates()
data_pat['DOB'] = pd.to_datetime(data_pat['DOB']).dt.date
data_pat['DOB'] = pd.to_datetime(data_pat['DOB'])
data_pat  = data_pat.drop_duplicates()
data_pat  = data_pat.reset_index(drop=True)

data_adm  = admissions[admissions['SUBJECT_ID'].isin(diabetic_sub_id)]
data_adm  = data_adm[['SUBJECT_ID','HADM_ID','ADMITTIME']]
data_adm['ADMITTIME'] = pd.to_datetime(data_adm['ADMITTIME']).dt.date
data_adm['ADMITTIME'] = pd.to_datetime(data_adm['ADMITTIME'])
data_adm  = data_adm.drop_duplicates()
data_adm  = data_adm.reset_index(drop=True)

label_file = label_file[label_file['SUBJECT_ID'].isin(diabetic_sub_id)]
  
def process_pres(data_pres):
    data_pres = data_pres.drop(columns=['ROW_ID','ICUSTAY_ID','STARTDATE','ENDDATE','DRUG_TYPE','DRUG','DRUG_NAME_POE','DRUG_NAME_GENERIC',
                     'FORMULARY_DRUG_CD','GSN','PROD_STRENGTH','DOSE_VAL_RX',
                     'DOSE_UNIT_RX','FORM_VAL_DISP','FORM_UNIT_DISP',
                      'ROUTE'], axis=1)
    data_pres.drop(index = data_pres[data_pres['NDC'] == '0'].index, axis=0, inplace=True)
    data_pres.dropna(inplace=True)
    data_pres.drop_duplicates(inplace=True)
    data_pres = data_pres.reset_index(drop=True)
    
    return data_pres


def ndc2atc4(data_pres):
    with open(ndc2rxnorm_file, 'r') as f:
        ndc2rxnorm = eval(f.read())
    data_pres['NDC'] = data_pres['NDC'].astype(str).apply(lambda x: x.zfill(11)) # written by yulong
    data_pres['RXCUI'] = data_pres['NDC'].map(ndc2rxnorm)
    data_pres.dropna(inplace=True)
    
    rxnorm2atc = pd.read_csv(ndc2atc_file)
    rxnorm2atc = rxnorm2atc.drop(columns=['YEAR','MONTH','NDC'])
    rxnorm2atc.drop_duplicates(subset=['RXCUI'], inplace=True)
    data_pres.drop(index = data_pres[data_pres['RXCUI'].isin([''])].index, axis=0, inplace=True)
    
    data_pres['RXCUI'] = data_pres['RXCUI'].astype('int64')
    data_pres = data_pres.reset_index(drop=True)
    data_pres = data_pres.merge(rxnorm2atc, on=['RXCUI'])
    data_pres.drop(columns=['NDC', 'RXCUI'], inplace=True)
    data_pres = data_pres.rename(columns={'ATC4':'NDC'})
    data_pres['NDC'] = data_pres['NDC'].map(lambda x: x[:4])
    data_pres = data_pres.drop_duplicates()    
    data_pres = data_pres.reset_index(drop=True)
    return data_pres

data_pres = prescriptions[prescriptions['SUBJECT_ID'].isin(diabetic_sub_id)]
data_pres = data_pres.reset_index(drop=True)
data_pres = process_pres(data_pres)
data_pres = ndc2atc4(data_pres)
data_pres['Count'] = 1
data_pres = data_pres.groupby(by=['SUBJECT_ID','HADM_ID','NDC'])['Count'].max().reset_index()
data_pres = pd.pivot_table(data_pres,index=['SUBJECT_ID','HADM_ID'],columns=['NDC'],values=['Count'])
data_pres.columns = data_pres.columns.droplevel()
data_pres.columns.name = None
data_pres = data_pres.reset_index()

# flatten and merge
final_data = data_pat.merge(data_adm, on='SUBJECT_ID', how='outer')
final_data['ADMITYEAR'] = final_data['ADMITTIME'].dt.year
final_data['DOB_YEAR'] = final_data['DOB'].dt.year
final_data['Age'] = final_data['ADMITYEAR'] - final_data['DOB_YEAR']
final_data.loc[final_data['Age']>100,'Age'] = 89
final_data        = final_data.drop(columns=['DOB'])
final_data.sort_values(by=['SUBJECT_ID','ADMITTIME'], inplace=True)
final_data = final_data.drop(columns=['ADMITYEAR', 'DOB_YEAR'], axis=1) # should consider religion, marital_status, ethnicity next time
final_data = final_data.drop_duplicates()

data_lab = pd.read_csv('D:\Work\mimic_iii_deephit_rmtpp\mapped_elements\CHARTEVENTS_reduced_24_hour_blocks_plus_admissions.csv')
data_lab = data_lab.drop(columns=['HADMID_DAY'])
data_lab = data_lab.groupby('HADM_ID').max().reset_index()
data_lab = data_lab.merge(admissions[['SUBJECT_ID','HADM_ID']].drop_duplicates(),on='HADM_ID',how='left')
final_data = final_data.merge(data_lab,on=['SUBJECT_ID','HADM_ID'],how='left')
final_data = final_data.merge(data_pres,on=['SUBJECT_ID','HADM_ID'],how='left')
cols_x = ['SUBJECT_ID', 'GENDER', 'HADM_ID', 'ADMITTIME', 'Age', 'BLACK']
cols_y = [y for y in final_data.columns if y not in cols_x]
cols = cols_x + cols_y
final_data = final_data[cols]

final_data = final_data.dropna(subset=final_data.columns[6:].tolist(),how='all')
final_data = final_data.sort_values(by=['SUBJECT_ID','ADMITTIME'])
final_data = final_data.merge(label_file,on=['SUBJECT_ID','HADM_ID'],how='left')

# drop unique rows containing subject_id. now each subject_id should have >= two case_visits
final_data = final_data[final_data.duplicated(subset=['SUBJECT_ID'],keep=False)]

final_data['LABEL_CVD_ALL'] = 0
final_data.loc[final_data['LABEL_MACE']==1,'LABEL_CVD_ALL'] = 1
final_data.loc[final_data['LABEL_CVD']==1,'LABEL_CVD_ALL'] = 1

# Adjust for repeated events please
# For those only got one label_mace, i will set the tte and choose randomly the number of visits.
# What about those who have multiple events? Definitely, I want to have more than one mace. 
# So what I will do is that I will choose randomly the number of events between the second mace and last mace.
# Then find the nearest mace event to the latest visit date.



event_boolean_label = final_data[['SUBJECT_ID','ADMITTIME','HADM_ID','LABEL_CVD_ALL']].drop_duplicates()
event_boolean_label = event_boolean_label.sort_values(by=['SUBJECT_ID','ADMITTIME'])
total_mace = event_boolean_label.groupby('SUBJECT_ID')['LABEL_CVD_ALL'].sum().reset_index()
patid_1mace = total_mace[total_mace['LABEL_CVD_ALL']==1]['SUBJECT_ID'].unique()
patid_mt2mace = total_mace[total_mace['LABEL_CVD_ALL']>1]['SUBJECT_ID'].unique()
patid_nomace = total_mace[total_mace['LABEL_CVD_ALL']==0]['SUBJECT_ID'].unique()

cnt = 0
final_datav2 = pd.DataFrame()
final_datav2_labelvec = pd.DataFrame()
patid_mt2mace_withlabel0 = []
for i_mt2mace in patid_mt2mace: # Have not remove those with tte = 0 yet
    cnt = cnt + 1
    if cnt % 100 == 0:
        print(cnt)
    temp = final_data[final_data['SUBJECT_ID']==i_mt2mace]
    if temp.iloc[len(temp)-1]['LABEL_CVD_ALL'] == 0:
        patid_mt2mace_withlabel0.append(i_mt2mace)    
    temp2 = temp.iloc[:-1]
    num2 = np.where(temp2['LABEL_CVD_ALL']==1)[0][-1] # index of visit where last mace occurs
    mthh_diff = (temp.iloc[len(temp)-1]['ADMITTIME'] - temp2.iloc[num2]['ADMITTIME'])/np.timedelta64(1,'M')
    mthh_diff = int(mthh_diff)    
    if mthh_diff > 0:
        final_datav2 = final_datav2.append(temp2.iloc[:(num2+1)])
        final_datav2_labelvec = final_datav2_labelvec.append(temp.iloc[(len(temp)-1):(len(temp))][['SUBJECT_ID','ADMITTIME']])
    while (mthh_diff < 1) & (num2 > 0):
        temp2 = temp2.iloc[:-1]
        if np.where(temp2['LABEL_CVD_ALL']==1)[0].size == 0 :
            num2 = len(temp2) - 1
        else:
            num2 = np.where(temp2['LABEL_CVD_ALL']==1)[0][-1] # index of visit where last mace occurs
        mthh_diff = (temp.iloc[len(temp)-1]['ADMITTIME'] - temp2.iloc[num2]['ADMITTIME'])/np.timedelta64(1,'M')
        mthh_diff = int(mthh_diff)            
        if mthh_diff > 0:
            final_datav2 = final_datav2.append(temp2.iloc[:(num2+1)])
            final_datav2_labelvec = final_datav2_labelvec.append(temp.iloc[(len(temp)-1):(len(temp))][['SUBJECT_ID','ADMITTIME']])

del mthh_diff
    
cnt = 0    
for i_1mace in patid_1mace:
    cnt = cnt + 1
    if cnt % 100 == 0:
        print(cnt)
    temp = final_data[final_data['SUBJECT_ID']==i_1mace]
    num1 = np.where(temp['LABEL_CVD_ALL']==1)[0][0] # index of visit where first mace occurs
    if num1 == 1:
        mth_diff = (temp.iloc[num1]['ADMITTIME'] - temp.iloc[num1-1]['ADMITTIME'])/np.timedelta64(1,'M')
        mth_diff = int(mth_diff)
        if mth_diff == 0:
            continue   
    if num1 == 0: # This automatically removes patients with first visit has mace, but it doesn't remove patients with mace that is within one month from 1st visit
        continue
    temp_nbrows = random.randint(0,num1-1) #randrange(num1) # Generate a random number from 0 to num1 - 1 #random.randint(0,num1-1)
    mthh_diff = (temp.iloc[num1]['ADMITTIME'] - temp.iloc[temp_nbrows]['ADMITTIME'])/np.timedelta64(1,'M')
    mthh_diff = int(mthh_diff)
    if mthh_diff > 0:
        final_datav2 = final_datav2.append(temp.iloc[:temp_nbrows+1])
        final_datav2_labelvec = final_datav2_labelvec.append(temp.iloc[num1: num1+1][['SUBJECT_ID','ADMITTIME']])  
    while (mthh_diff < 1) & (temp_nbrows > 0):
        temp_nbrows = temp_nbrows - 1
        mthh_diff = (temp.iloc[num1]['ADMITTIME'] - temp.iloc[temp_nbrows]['ADMITTIME'])/np.timedelta64(1,'M')
        mthh_diff = int(mthh_diff)
        if mthh_diff > 0:
            final_datav2 = final_datav2.append(temp.iloc[:temp_nbrows+1])
            final_datav2_labelvec = final_datav2_labelvec.append(temp.iloc[num1: num1+1][['SUBJECT_ID','ADMITTIME']])                  

del mthh_diff

cnt = 0
for i_nomace in patid_nomace:
    cnt = cnt + 1
    if cnt % 100 == 0:
        print(cnt)
    temp = final_data[final_data['SUBJECT_ID']==i_nomace]
    num0 = len(temp) - 1
    if num0 == 1:
        mthhh_diff = (temp.iloc[num0]['ADMITTIME'] - temp.iloc[num0-1]['ADMITTIME'])/np.timedelta64(1,'M')
        mthhh_diff = int(mthhh_diff)
        if mthhh_diff == 0:
            continue
    if num0 == 0:
        continue
    temp_nbrows = random.randint(0,num0-1) #randrange(num1) # Generate a random number from 0 to num1 - 1 #random.randint(0,num1-1)
    mthhh_diff = (temp.iloc[num0]['ADMITTIME'] - temp.iloc[temp_nbrows]['ADMITTIME'])/np.timedelta64(1,'M')
    mthhh_diff = int(mthhh_diff)
    if mthhh_diff > 0:
        final_datav2 = final_datav2.append(temp.iloc[:temp_nbrows+1])
        final_datav2_labelvec = final_datav2_labelvec.append(temp.iloc[num0: num0+1][['SUBJECT_ID','ADMITTIME']])          
    while (mthhh_diff < 1) & (temp_nbrows > 0):
        temp_nbrows = temp_nbrows - 1
        mthhh_diff = (temp.iloc[num0]['ADMITTIME'] - temp.iloc[temp_nbrows]['ADMITTIME'])/np.timedelta64(1,'M')
        mthhh_diff = int(mthhh_diff)
        if mthhh_diff > 0:
            final_datav2 = final_datav2.append(temp.iloc[:temp_nbrows+1])
            final_datav2_labelvec = final_datav2_labelvec.append(temp.iloc[num0: num0+1][['SUBJECT_ID','ADMITTIME']])                      
        
final_datav2_orig = copy.deepcopy(final_datav2)
final_datav2_labelvec_orig = copy.deepcopy(final_datav2_labelvec)
            
#final_datav2 = final_datav2.append(final_data[final_data['SUBJECT_ID'].isin(patid_nomace)])

final_datav2_labelvec = final_datav2_labelvec[['SUBJECT_ID','ADMITTIME']].drop_duplicates()
aa = copy.deepcopy(final_datav2_labelvec)
aa = aa.rename(columns={'ADMITTIME':'dateCC'})
aa.loc[aa['SUBJECT_ID'].isin(patid_mt2mace_withlabel0+list(patid_nomace)),'dateCC'] = np.nan
final_datav2_labelvec = final_datav2_labelvec.rename(columns={'ADMITTIME':'LAST_VISIT'})
final_datav2_labelvec = final_datav2_labelvec.merge(aa,on=['SUBJECT_ID'],how='left')


final_datav2 = final_datav2.drop(columns=['LABEL_MACE','LABEL_DIABETES','LABEL_CVD_ALL','LABEL_CVD'])
final_datav2 = final_datav2.merge(final_datav2_labelvec,on=['SUBJECT_ID'],how='left')

data_final = copy.deepcopy(final_datav2)
cols_x = ['SUBJECT_ID', 'ADMITTIME', 'HADM_ID', 'dateCC','LAST_VISIT']
cont_col = ['Age','ACR', 'ALT', 'AST', 'Albumin', 'CK', 'Creatinine', 'GGT', 'Glucose',
             'HDL', 'HbA1c', 'Height', 'LDL', 'NT-PROBNP', 'Platelets', 'Potassium',
              'Protein Creatinine Ratio', 'Total', 'Triglycerides', 'Urea', 'daily weight',
               'diastolic', 'heart rate', 'systolic']

bin_col = [y for y in data_final.columns if y not in cols_x+cont_col]
bin_col = bin_col[2:] + bin_col[0:2]
cols = cols_x + cont_col + bin_col
data_final = data_final[cols]
data_final['GENDER'] = data_final['GENDER'].replace({'M':0,'F':1})
data_final[bin_col] = data_final[bin_col].fillna(0)
data_final = data_final.sort_values(by=['SUBJECT_ID','ADMITTIME'])

data_first = data_final[['SUBJECT_ID','ADMITTIME']].groupby(by=['SUBJECT_ID']).first().reset_index()
data_first = data_first.rename(columns={'ADMITTIME':'FIRST_VISIT'})

data_final['Label'] = 0
data_final.loc[pd.notnull(data_final['dateCC']),'Label']=1
data_final = data_final.merge(data_first,on='SUBJECT_ID',how='left')

data_final['tte'] = ((data_final['LAST_VISIT']-data_final['FIRST_VISIT'])/np.timedelta64(1,'M'))
data_final['tte'] = data_final['tte'].astype(int) # Need to remove those with tte<0 becos they dun have any phy,lab,med info in visits before complication occured.

data_final['times'] = ((data_final['ADMITTIME']-data_final['FIRST_VISIT'])/np.timedelta64(1,'M'))
data_final['times'] = data_final['times'].astype(int)

cols_x = ['SUBJECT_ID','ADMITTIME','HADM_ID','FIRST_VISIT', 'LAST_VISIT','dateCC','times','tte','Label']
cols_y = [y for y in data_final.columns if y not in cols_x]
cols = cols_x + cols_y
data_final = data_final[cols]
data_final = data_final.merge(event_boolean_label[['SUBJECT_ID','HADM_ID','LABEL_CVD_ALL']],on=['SUBJECT_ID','HADM_ID'],how='left')
data_final = data_final.rename(columns={'LABEL_CVD_ALL':'EVENT_BOOLEAN'})
data_final_orig = copy.deepcopy(data_final)

retain_nric = data_final[data_final['tte']>0]['SUBJECT_ID'].unique() # Why is this not removed already i_1mace?
# Reason: # patid_nomace, their last visit - first visit is really tte=0
#       : # patid_mt2mace, their last mace from the previous mace is really tte=0, readmission with condition mace is within a month, somthimes within a few days
#       : # patid_1mace, For patients with only one mace, we can control to prevent them from having tte = 0 provided they have more than 2 visits.

data_final = data_final[data_final['SUBJECT_ID'].isin(retain_nric)]

check_nric = data_final[data_final['times'] >= data_final['tte']]['SUBJECT_ID'].unique()
data_final = data_final[data_final['times'] < data_final['tte']]

#mace_boolean = label_file[label_file['LABEL_MACE']==1]['HADM_ID'].unique()
#cvd_boolean = label_file[label_file['LABEL_CVD']==1]['HADM_ID'].unique()
#data_final.loc[data_final['HADM_ID'].isin(cvd_boolean),'EVENT_BOOLEAN'] = 2 
# EVENT_BOOLEAN equal to marker: 1 for mace, 2 for cvd only

for i in check_nric:
    event_nb = data_final[data_final['SUBJECT_ID']==i][['ADMITTIME','dateCC','times','tte','EVENT_BOOLEAN']]
    print(event_nb)
    input('pause')


rmtpp_data = data_final[data_final['EVENT_BOOLEAN']>0][['SUBJECT_ID','times','EVENT_BOOLEAN']]
data_tgt = data_final[data_final['EVENT_BOOLEAN']>0][['SUBJECT_ID','tte','LAST_VISIT']].drop_duplicates()    
data_tgt = data_tgt.merge(final_data[['SUBJECT_ID','ADMITTIME','LABEL_CVD','LABEL_MACE']],left_on=['SUBJECT_ID','LAST_VISIT'],right_on=['SUBJECT_ID','ADMITTIME'],how='left')
data_tgt['EVENT_BOOLEAN'] = 0 # Those with mace event is label as 0
data_tgt.loc[data_tgt['LABEL_CVD']==1,'EVENT_BOOLEAN'] = 1 # THose with cvd event is label as 1
data_tgt = data_tgt.drop(columns=['LAST_VISIT','ADMITTIME','LABEL_CVD','LABEL_MACE'])
data_tgt = data_tgt.rename(columns={'tte':'times'})
final_rmtpp_data = pd.concat([rmtpp_data,data_tgt],axis=0)
final_rmtpp_data = final_rmtpp_data.sort_values(by=['SUBJECT_ID','times','EVENT_BOOLEAN'])
final_rmtpp_data = final_rmtpp_data.rename(columns={'SUBJECT_ID':'id','times':'time','EVENT_BOOLEAN':'event'})

# Fix seed
np.random.seed(np_rand_seed) # Need to set for every run to obtain fixed pos_NRICperm, neg_NRICperm
pos_len = len(final_rmtpp_data['id'].unique())
pos_NRIC = final_rmtpp_data['id'].unique().tolist()
pos_NRICperm = np.random.permutation(pos_NRIC)
cross_ratio = 0.2
set1_pos_index = range(int(round(cross_ratio*pos_len)))
set2_pos_index = range(int(round(cross_ratio*pos_len)))
set3_pos_index = range(int(round(cross_ratio*pos_len)))
set4_pos_index = range(int(round(cross_ratio*pos_len)))
set5_pos_index = range(pos_len-len(set1_pos_index)-len(set2_pos_index)-len(set3_pos_index)-len(set4_pos_index))
set2_pos_index = [x+len(set1_pos_index) for x in set2_pos_index]
set3_pos_index = [x+len(set1_pos_index)+len(set2_pos_index) for x in set3_pos_index]
set4_pos_index = [x+len(set1_pos_index)+len(set2_pos_index)+len(set3_pos_index) for x in set4_pos_index]
set5_pos_index = [x+len(set1_pos_index)+len(set2_pos_index)+len(set3_pos_index)+len(set4_pos_index) for x in set5_pos_index]

train_nric1 = np.concatenate((pos_NRICperm[set1_pos_index],
                              pos_NRICperm[set2_pos_index],
                              pos_NRICperm[set3_pos_index]))
valid_nric1 = pos_NRICperm[set4_pos_index]
test_nric1 = pos_NRICperm[set5_pos_index]

train_nric2 = np.concatenate((pos_NRICperm[set1_pos_index],
                              pos_NRICperm[set2_pos_index],
                              pos_NRICperm[set5_pos_index]))
valid_nric2 = pos_NRICperm[set3_pos_index]
test_nric2 = pos_NRICperm[set4_pos_index]

train_nric3 = np.concatenate((pos_NRICperm[set1_pos_index],
                              pos_NRICperm[set4_pos_index],
                              pos_NRICperm[set5_pos_index],))
valid_nric3 = pos_NRICperm[set2_pos_index] 
test_nric3 = pos_NRICperm[set3_pos_index]

train_nric4 = np.concatenate((pos_NRICperm[set3_pos_index],
                              pos_NRICperm[set4_pos_index],
                              pos_NRICperm[set5_pos_index]))
valid_nric4 = pos_NRICperm[set1_pos_index]
test_nric4 = pos_NRICperm[set2_pos_index]

train_nric5 = np.concatenate((pos_NRICperm[set2_pos_index],
                              pos_NRICperm[set3_pos_index],
                              pos_NRICperm[set4_pos_index]))
valid_nric5 = pos_NRICperm[set5_pos_index]
test_nric5 = pos_NRICperm[set1_pos_index]


final_rmtpp_data[final_rmtpp_data['id'].isin(train_nric1)].to_csv('D:\\Work\\mimic_iii_deephit_rmtpp\\ERPP-RMTPP-master_edited\\data\\folder1\\train_day.csv',index=False)
final_rmtpp_data[final_rmtpp_data['id'].isin(valid_nric1)].to_csv('D:\\Work\\mimic_iii_deephit_rmtpp\\ERPP-RMTPP-master_edited\\data\\folder1\\valid_day.csv',index=False)
final_rmtpp_data[final_rmtpp_data['id'].isin(test_nric1)].to_csv('D:\\Work\\mimic_iii_deephit_rmtpp\\ERPP-RMTPP-master_edited\\data\\folder1\\test_day.csv',index=False)

final_rmtpp_data[final_rmtpp_data['id'].isin(train_nric2)].to_csv('D:\\Work\\mimic_iii_deephit_rmtpp\\ERPP-RMTPP-master_edited\\data\\folder2\\train_day.csv',index=False)
final_rmtpp_data[final_rmtpp_data['id'].isin(valid_nric2)].to_csv('D:\\Work\\mimic_iii_deephit_rmtpp\\ERPP-RMTPP-master_edited\\data\\folder2\\valid_day.csv',index=False)
final_rmtpp_data[final_rmtpp_data['id'].isin(test_nric2)].to_csv('D:\\Work\\mimic_iii_deephit_rmtpp\\ERPP-RMTPP-master_edited\\data\\folder2\\test_day.csv',index=False)

final_rmtpp_data[final_rmtpp_data['id'].isin(train_nric3)].to_csv('D:\\Work\\mimic_iii_deephit_rmtpp\\ERPP-RMTPP-master_edited\\data\\folder3\\train_day.csv',index=False)
final_rmtpp_data[final_rmtpp_data['id'].isin(valid_nric3)].to_csv('D:\\Work\\mimic_iii_deephit_rmtpp\\ERPP-RMTPP-master_edited\\data\\folder3\\valid_day.csv',index=False)
final_rmtpp_data[final_rmtpp_data['id'].isin(test_nric3)].to_csv('D:\\Work\\mimic_iii_deephit_rmtpp\\ERPP-RMTPP-master_edited\\data\\folder3\\test_day.csv',index=False)

final_rmtpp_data[final_rmtpp_data['id'].isin(train_nric4)].to_csv('D:\\Work\\mimic_iii_deephit_rmtpp\\ERPP-RMTPP-master_edited\\data\\folder4\\train_day.csv',index=False)
final_rmtpp_data[final_rmtpp_data['id'].isin(valid_nric4)].to_csv('D:\\Work\\mimic_iii_deephit_rmtpp\\ERPP-RMTPP-master_edited\\data\\folder4\\valid_day.csv',index=False)
final_rmtpp_data[final_rmtpp_data['id'].isin(test_nric4)].to_csv('D:\\Work\\mimic_iii_deephit_rmtpp\\ERPP-RMTPP-master_edited\\data\\folder4\\test_day.csv',index=False)

final_rmtpp_data[final_rmtpp_data['id'].isin(train_nric5)].to_csv('D:\\Work\\mimic_iii_deephit_rmtpp\\ERPP-RMTPP-master_edited\\data\\folder5\\train_day.csv',index=False)
final_rmtpp_data[final_rmtpp_data['id'].isin(valid_nric5)].to_csv('D:\\Work\\mimic_iii_deephit_rmtpp\\ERPP-RMTPP-master_edited\\data\\folder5\\valid_day.csv',index=False)
final_rmtpp_data[final_rmtpp_data['id'].isin(test_nric5)].to_csv('D:\\Work\\mimic_iii_deephit_rmtpp\\ERPP-RMTPP-master_edited\\data\\folder5\\test_day.csv',index=False)

# Fix seed
np.random.seed(np_rand_seed) # Need to set for every run to obtain fixed pos_NRICperm, neg_NRICperm

# Randomize train, test and validation index [60%, 20%, 20%]
train_ratio = 0.6
test_ratio = 0.2
valid_ratio = 0.2

label_vec = data_final[['SUBJECT_ID','Label']].drop_duplicates()
pos_index = label_vec.index[label_vec['Label']==1].tolist()
neg_index = label_vec.index[label_vec['Label']==0].tolist()

pos_data = label_vec.loc[pos_index]
neg_data = label_vec.loc[neg_index]

pos_NRIC = pos_data['SUBJECT_ID'].unique().tolist()
neg_NRIC = neg_data['SUBJECT_ID'].unique().tolist()
pos_NRICperm = np.random.permutation(pos_NRIC)
neg_NRICperm = np.random.permutation(neg_NRIC)
pos_len = len(pos_NRICperm)
neg_len = len(neg_NRICperm)

train_pos_index = range(int(round(train_ratio*pos_len)))
valid_pos_index = range(int(round(valid_ratio*pos_len)))
valid_pos_index = [x+len(train_pos_index) for x in valid_pos_index]
test_pos_index = range(pos_len-len(train_pos_index)-len(valid_pos_index))
test_pos_index = [x+len(train_pos_index)+len(valid_pos_index) for x in test_pos_index]
train_neg_index = range(int(round(train_ratio*neg_len)))
valid_neg_index = range(int(round(valid_ratio*neg_len)))
valid_neg_index = [x+len(train_neg_index) for x in valid_neg_index]
test_neg_index = range(neg_len-len(train_neg_index)-len(valid_neg_index))
test_neg_index = [x+len(train_neg_index)+len(valid_neg_index) for x in test_neg_index]

train_nric = np.concatenate((pos_NRICperm[train_pos_index],neg_NRICperm[train_neg_index]))
valid_nric = np.concatenate((pos_NRICperm[valid_pos_index],neg_NRICperm[valid_neg_index]))
test_nric = np.concatenate((pos_NRICperm[test_pos_index],neg_NRICperm[test_neg_index]))

cross_ratio = 0.2
set1_pos_index = range(int(round(cross_ratio*pos_len)))
set2_pos_index = range(int(round(cross_ratio*pos_len)))
set3_pos_index = range(int(round(cross_ratio*pos_len)))
set4_pos_index = range(int(round(cross_ratio*pos_len)))
set5_pos_index = range(pos_len-len(set1_pos_index)-len(set2_pos_index)-len(set3_pos_index)-len(set4_pos_index))
set2_pos_index = [x+len(set1_pos_index) for x in set2_pos_index]
set3_pos_index = [x+len(set1_pos_index)+len(set2_pos_index) for x in set3_pos_index]
set4_pos_index = [x+len(set1_pos_index)+len(set2_pos_index)+len(set3_pos_index) for x in set4_pos_index]
set5_pos_index = [x+len(set1_pos_index)+len(set2_pos_index)+len(set3_pos_index)+len(set4_pos_index) for x in set5_pos_index]

set1_neg_index = range(int(round(cross_ratio*neg_len)))
set2_neg_index = range(int(round(cross_ratio*neg_len)))
set3_neg_index = range(int(round(cross_ratio*neg_len)))
set4_neg_index = range(int(round(cross_ratio*neg_len)))
set5_neg_index = range(neg_len-len(set1_neg_index)-len(set2_neg_index)-len(set3_neg_index)-len(set4_neg_index))
set2_neg_index = [x+len(set1_neg_index) for x in set2_neg_index]
set3_neg_index = [x+len(set1_neg_index)+len(set2_neg_index) for x in set3_neg_index]
set4_neg_index = [x+len(set1_neg_index)+len(set2_neg_index)+len(set3_neg_index) for x in set4_neg_index]
set5_neg_index = [x+len(set1_neg_index)+len(set2_neg_index)+len(set3_neg_index)+len(set4_neg_index) for x in set5_neg_index]

train_nric1 = np.concatenate((pos_NRICperm[set1_pos_index],neg_NRICperm[set1_neg_index],
                              pos_NRICperm[set2_pos_index],neg_NRICperm[set2_neg_index],
                              pos_NRICperm[set3_pos_index],neg_NRICperm[set3_neg_index]))
valid_nric1 = np.concatenate((pos_NRICperm[set4_pos_index],neg_NRICperm[set4_neg_index]))
test_nric1 = np.concatenate((pos_NRICperm[set5_pos_index],neg_NRICperm[set5_neg_index]))

train_nric2 = np.concatenate((pos_NRICperm[set1_pos_index],neg_NRICperm[set1_neg_index],
                              pos_NRICperm[set2_pos_index],neg_NRICperm[set2_neg_index],
                              pos_NRICperm[set5_pos_index],neg_NRICperm[set5_neg_index]))
valid_nric2 = np.concatenate((pos_NRICperm[set3_pos_index],neg_NRICperm[set3_neg_index]))
test_nric2 = np.concatenate((pos_NRICperm[set4_pos_index],neg_NRICperm[set4_neg_index]))

train_nric3 = np.concatenate((pos_NRICperm[set1_pos_index],neg_NRICperm[set1_neg_index],
                              pos_NRICperm[set4_pos_index],neg_NRICperm[set4_neg_index],
                              pos_NRICperm[set5_pos_index],neg_NRICperm[set5_neg_index]))
valid_nric3 = np.concatenate((pos_NRICperm[set2_pos_index],neg_NRICperm[set2_neg_index]))
test_nric3 = np.concatenate((pos_NRICperm[set3_pos_index],neg_NRICperm[set3_neg_index]))

train_nric4 = np.concatenate((pos_NRICperm[set3_pos_index],neg_NRICperm[set3_neg_index],
                              pos_NRICperm[set4_pos_index],neg_NRICperm[set4_neg_index],
                              pos_NRICperm[set5_pos_index],neg_NRICperm[set5_neg_index]))
valid_nric4 = np.concatenate((pos_NRICperm[set1_pos_index],neg_NRICperm[set1_neg_index]))
test_nric4 = np.concatenate((pos_NRICperm[set2_pos_index],neg_NRICperm[set2_neg_index]))

train_nric5 = np.concatenate((pos_NRICperm[set2_pos_index],neg_NRICperm[set2_neg_index],
                              pos_NRICperm[set3_pos_index],neg_NRICperm[set3_neg_index],
                              pos_NRICperm[set4_pos_index],neg_NRICperm[set4_neg_index]))
valid_nric5 = np.concatenate((pos_NRICperm[set5_pos_index],neg_NRICperm[set5_neg_index]))
test_nric5 = np.concatenate((pos_NRICperm[set1_pos_index],neg_NRICperm[set1_neg_index]))

list1 = data_final['SUBJECT_ID'].unique().tolist()
list2 = list(range(1, len(list1)+1))
id_df = pd.DataFrame(zip(list1,list2),columns=['SUBJECT_ID','id'])
pbc_cleaned = data_final.merge(id_df,on='SUBJECT_ID',how='left')
pbc_cleaned = pbc_cleaned.rename(columns={'Label':'label'})
label_vec = pbc_cleaned[['SUBJECT_ID','label','id']].drop_duplicates()
pbc_cleaned = pbc_cleaned.drop(columns=['SUBJECT_ID', 'ADMITTIME', 'HADM_ID', 'FIRST_VISIT', 'LAST_VISIT','dateCC'])                     
cols_x = ['id','tte','times','label']
cols_y = [y for y in pbc_cleaned.columns if y not in cols_x]
cols = cols_x + cols_y
pbc_cleaned = pbc_cleaned[cols]        

# Remove columns with zeros
temp_pbc = copy.deepcopy(pbc_cleaned)
temp_pbc = temp_pbc.fillna(0)
temp_pbc = np.array(temp_pbc)
zero_index_column = list(np.where(np.sum(temp_pbc,axis=0)==0)[0])
pbc_cleaned.drop(pbc_cleaned.columns[zero_index_column],axis=1,inplace=True)

pbc_cleaned.to_csv(os.path.join(workdir,'Dynamic-DeepHit-master/data','mimic_mace_cvdrepeatedevents.csv'),index=False)
with open(os.path.join(workdir,'Dynamic-DeepHit-master/data','mimic_mace_cvdrepeatedevents_label.pickle'),'wb') as f:
    pickle.dump([label_vec,train_nric1,valid_nric1,test_nric1,
                 train_nric2,valid_nric2,test_nric2,
                 train_nric3,valid_nric3,test_nric3,
                 train_nric4,valid_nric4,test_nric4,
                 train_nric5,valid_nric5,test_nric5],f)
    
    
    
# Plot Histogram of repeated events
from collections import Counter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
cvdrepeatedevents = pd.read_csv('D:\Work\mimic_iii_deephit_rmtpp\Dynamic-DeepHit-master\data\mimic_mace_cvdrepeatedevents.csv')
repeatedcvd = cvdrepeatedevents.groupby('id')['EVENT_BOOLEAN'].sum().reset_index()
counter_obj = Counter(repeatedcvd['EVENT_BOOLEAN'])
objects = ('0','1','2','3','4','>4')
y_pos = [0,1,2,3,4,5]
percentage = [counter_obj[0],counter_obj[1],counter_obj[2],counter_obj[3],counter_obj[4], sum(counter_obj.values()) - counter_obj[0]-counter_obj[1]-counter_obj[2]-counter_obj[3]-counter_obj[4]]
percentage = list(np.array(percentage)/sum(counter_obj.values())*100)

plt.bar(y_pos,percentage, align='center',alpha=0.5, width = 0.6, edgecolor='b',linewidth=1)
plt.xticks(y_pos,objects)
plt.ylabel('% patients')
plt.xlabel('Number of CVD events to a patient')
plt.ylim(0,100)
ax = plt.gca()
ax.yaxis.grid()


    