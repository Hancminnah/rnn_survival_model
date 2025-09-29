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
import math
import multiprocessing as mp
from time import sleep

np_rand_seed = 42

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

label_vec = final_data[['SUBJECT_ID','ADMITTIME','LABEL_CVD_ALL']].drop_duplicates()
label_vec = label_vec.sort_values(by=['SUBJECT_ID','ADMITTIME'])
label_vec = label_vec[label_vec['LABEL_CVD_ALL'] == 1]
label_vec = label_vec.drop(columns=['LABEL_CVD_ALL'])
label_vec = label_vec.groupby('SUBJECT_ID').first().reset_index()
label_vec = label_vec.rename(columns={'ADMITTIME':'dateCC'})
final_data = final_data.drop(columns=['LABEL_MACE','LABEL_DIABETES','LABEL_CVD','LABEL_CVD_ALL'])
final_data = final_data.merge(label_vec,on=['SUBJECT_ID'],how='left')

data_final = copy.deepcopy(final_data)
cols_x = ['SUBJECT_ID', 'ADMITTIME', 'HADM_ID', 'dateCC']
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

data_censored = data_final[pd.isnull(data_final['dateCC'])][['SUBJECT_ID','ADMITTIME']].drop_duplicates()
data_censored_last = data_censored.groupby(by=['SUBJECT_ID']).last().reset_index()
data_censored_last = data_censored_last.rename(columns={'ADMITTIME':'LAST_VISIT'})
data_censored_last['Label'] = 0

data_uncensored_last = data_final[pd.notnull(data_final['dateCC'])][['SUBJECT_ID','dateCC']].drop_duplicates()
data_uncensored_last = data_uncensored_last.rename(columns={'dateCC':'LAST_VISIT'})
data_uncensored_last['Label'] = 1
data_last = pd.concat([data_censored_last,data_uncensored_last],axis=0)
data_first_last = data_first.merge(data_last,on='SUBJECT_ID',how='left')
data_first_last['tte'] = ((data_first_last['LAST_VISIT']-data_first_last['FIRST_VISIT'])/np.timedelta64(1,'M'))
data_first_last['tte'] = data_first_last['tte'].astype(int) # Need to remove those with tte<0 becos they dun have any phy,lab,med info in visits before complication occured.

data_final = data_final.merge(data_first,on='SUBJECT_ID',how='left')
data_final['times'] = ((data_final['ADMITTIME']-data_final['FIRST_VISIT'])/np.timedelta64(1,'M'))
data_final['times'] = data_final['times'].astype(int)
data_final = data_final.drop(columns=['FIRST_VISIT'])
data_final = data_final.merge(data_first_last,on='SUBJECT_ID',how='left')

cols_x = ['SUBJECT_ID','ADMITTIME','HADM_ID','FIRST_VISIT', 'LAST_VISIT','dateCC','times','tte','Label']
cols_y = [y for y in data_final.columns if y not in cols_x]
cols = cols_x + cols_y
data_final = data_final[cols]
data_final_orig = copy.deepcopy(data_final)

retain_nric = data_final[data_final['tte']>0]['SUBJECT_ID'].unique()
data_final = data_final[data_final['SUBJECT_ID'].isin(retain_nric)]

data_final = data_final[data_final['times'] < data_final['tte']]

#data_finalv2 = pd.DataFrame()
#for i in data_final['SUBJECT_ID'].unique():
#    temp = data_final[data_final['SUBJECT_ID']==i]
#    temp_nbrows = random.randint(1,len(temp)-1)
#    data_finalv2 = data_finalv2.append(temp.iloc[:temp_nbrows])
#data_finalv2 = data_finalv2.sort_values(by=['SUBJECT_ID','ADMITTIME'])


data_finalv2 = pd.DataFrame()
cnt = 0
for i in data_final['SUBJECT_ID'].unique():
    cnt = cnt + 1
    if cnt % 1000 == 0:
        print(str(cnt) + ' out of ' + str(len(data_final['SUBJECT_ID'].unique())))
    temp = data_final[data_final['SUBJECT_ID']==i]
    num_all = len(temp) - 1
    temp_nbrows = random.randint(0,num_all)
    mth_diff = (temp.iloc[num_all]['LAST_VISIT']-temp.iloc[temp_nbrows]['ADMITTIME'])/np.timedelta64(1,'M')
    mth_diff = int(mth_diff)
    if mth_diff > 0:
        data_finalv2 = data_finalv2.append(temp.iloc[:temp_nbrows+1])
    while (mth_diff < 1) & (temp_nbrows >0):
           temp_nbrows = temp_nbrows - 1
           mth_diff = (temp.iloc[num_all]['LAST_VISIT']-temp.iloc[temp_nbrows]['ADMITTIME'])/np.timedelta64(1,'M')
           mth_diff = int(mth_diff)
           if mth_diff > 0:
               data_finalv2 = data_finalv2.append(temp.iloc[:temp_nbrows+1])
               
# Fix seed
np.random.seed(np_rand_seed) # Need to set for every run to obtain fixed pos_NRICperm, neg_NRICperm

# Randomize train, test and validation index [60%, 20%, 20%]
train_ratio = 0.6
test_ratio = 0.2
valid_ratio = 0.2

label_vec = data_finalv2[['SUBJECT_ID','Label']].drop_duplicates()
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

list1 = data_finalv2['SUBJECT_ID'].unique().tolist()
list2 = list(range(1, len(list1)+1))
id_df = pd.DataFrame(zip(list1,list2),columns=['SUBJECT_ID','id'])
pbc_cleaned = data_finalv2.merge(id_df,on='SUBJECT_ID',how='left')
pbc_cleaned = pbc_cleaned.rename(columns={'Label':'label'})
label_vec = pbc_cleaned[['SUBJECT_ID','label','id']].drop_duplicates()
pbc_cleaned = pbc_cleaned.drop(columns=['SUBJECT_ID', 'ADMITTIME', 'HADM_ID', 'FIRST_VISIT', 'LAST_VISIT','dateCC'])                     
cols_x = ['id','tte','times','label']
cols_y = [y for y in pbc_cleaned.columns if y not in cols_x]
cols = cols_x + cols_y
pbc_cleaned = pbc_cleaned[cols]        
   
pbc_cleaned.to_csv(os.path.join(workdir,'Dynamic-DeepHit-master/data','mimic_mace_cvd.csv'),index=False)
with open(os.path.join(workdir,'Dynamic-DeepHit-master/data','mimic_mace_label_cvd.pickle'),'wb') as f:
    pickle.dump([label_vec,train_nric1,valid_nric1,test_nric1,
                 train_nric2,valid_nric2,test_nric2,
                 train_nric3,valid_nric3,test_nric3,
                 train_nric4,valid_nric4,test_nric4,
                 train_nric5,valid_nric5,test_nric5],f)
    
    
    
    
    