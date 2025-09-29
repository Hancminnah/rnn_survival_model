_EPSILON = 1e-08

import numpy as np
import pandas as pd
import tensorflow as tf
import random
import os
import pickle
import bz2

from sklearn.model_selection import train_test_split
from scipy.integrate import trapz, quad
from sklearn.metrics import mean_squared_error
from lifelines.utils import concordance_index

import import_data_rmtpp_repeated_events as impt

from class_DeepLongitudinal_final_repeatedevents import Model_Longitudinal_Attention

from utils_eval             import c_index, brier_score
from utils_log              import save_logging, load_logging
from utils_helper_v2           import f_get_minibatch, f_get_boosted_trainset

from tensorflow.contrib.layers import fully_connected as FC_Net
from tensorflow.python.ops.rnn import _transpose_batch_time

import utils_network as utils

fload = 1
comp_str = 'MACE'
set_str = 1
# In[ ]:


def _f_get_pred(sess, model, data, data_mi, pred_horizon):
    '''
        predictions based on the prediction time.
        create new_data and new_mask2 that are available previous or equal to the prediction time (no future measurements are used)
    '''
    new_data    = np.zeros(np.shape(data))
    new_data_mi = np.zeros(np.shape(data_mi))

    meas_time = np.concatenate([np.zeros([np.shape(data)[0], 1]), np.cumsum(data[:, :, 0], axis=1)[:, :-1]], axis=1)

    for i in range(np.shape(data)[0]):
        last_meas = np.sum(meas_time[i, :] <= pred_horizon)

        new_data[i, :last_meas, :]    = data[i, :last_meas, :]
        new_data_mi[i, :last_meas, :] = data_mi[i, :last_meas, :]

    return model.predict(new_data, new_data_mi)


def f_get_risk_predictions(sess, model, data_, data_mi_, pred_time, eval_time, new_par):
    
    pred = _f_get_pred(sess, model, data_[[0]], data_mi_[[0]], 0)
    _, num_Event = np.shape(pred)
       
    risk_all = {}
    for k in range(num_Event):
        risk_all[k] = np.zeros([np.shape(data_)[0], len(pred_time), len(eval_time)])
            
    for p, p_time in enumerate(pred_time):
        ### PREDICTION
        pred_horizon = int(p_time)
        pred = _f_get_pred(sess, model, data_, data_mi_, pred_horizon)


        for t, t_time in enumerate(eval_time):

            # calculate F(t | x, Y, t >= t_M) = \sum_{t_M <= \tau < t} P(\tau | x, Y, \tau > t_M)
            delta_t = int(t_time)/new_par['time_scale'] # divide by a certain timescale
            risk = np.exp((1/new_par['delta'])*np.exp(pred) - (1/new_par['delta'])*np.exp(pred+new_par['delta']*delta_t))
            risk = 1 - risk
            
            for k in range(num_Event):
                risk_all[k][:, p, t] = risk[:, k]
                
    return risk_all

def pred_next_eventtime(sess, model, data_, data_mi_, data_label, data_tgt, pred_time, new_par, nb_years):
    pred = _f_get_pred(sess, model, data_[[0]], data_mi_[[0]], 0)
    _, num_Event = np.shape(pred)

    next_event_time = {}
    rmse_months = {}

    for k in range(num_Event):
        next_event_time[k] = np.zeros([np.shape(data_)[0], len(pred_time), 1])
        rmse_months[k] = np.zeros([len(pred_time), 1])

    for p, p_time in enumerate(pred_time):
        pred_horizon = int(p_time)
        pred = _f_get_pred(sess, model, data_, data_mi_, pred_horizon)

        ts = np.arange(0, 12*nb_years*(1/new_par['time_scale']), 1/new_par['time_scale'])
        fs = np.exp(pred+new_par['delta']*ts + 1/new_par['delta']*np.exp(pred)-1/new_par['delta']*np.exp(pred+new_par['delta']*ts))
        df = ts * fs

        pred_next_eventtime_vec = trapz(df, ts)
        pred_next_eventtime_vec = pred_next_eventtime_vec * new_par['time_scale']

        for k in range(num_Event):
            next_event_time[k][:, p, 0] = pred_next_eventtime_vec
            retain_indices = np.where(data_label==1)[0]
            a = pred_next_eventtime_vec * np.squeeze(data_label)
            b = np.squeeze(data_tgt) * np.squeeze(data_label)
            rmse_months[k][p,0] = np.sqrt(mean_squared_error(a[retain_indices], b[retain_indices]))

    return next_event_time, rmse_months

# ### 1. Import Dataset
# #####      - Users must prepare dataset in csv format and modify 'import_data.py' following our examplar 'PBC2'

# In[ ]:


data_mode                   = comp_str + '_rmtpp_repeatedevents_set' + str(set_str)
seed                        = 1234

##### IMPORT DATASET
'''
    num_Category            = max event/censoring time * 1.2
    num_Event               = number of evetns i.e. len(np.unique(label))-1
    max_length              = maximum number of measurements
    x_dim                   = data dimension including delta (1 + num_features)
    x_dim_cont              = dim of continuous features
    x_dim_bin               = dim of binary features
    mask1, mask2, mask3     = used for cause-specific network (FCNet structure)
'''


if fload == 0:
    (x_dim, x_dim_cont, x_dim_bin), (data, time, label), (mask1, mask2, mask3), (data_mi), target = impt.import_dataset('./data/mimic_'+comp_str+'_cvdrepeatedevents.csv', norm_mode = 'standard')
    TUPLE3 = (x_dim, x_dim_cont, x_dim_bin, data, time, label, mask1, mask2, mask3, data_mi,target)
else:
    with open('mimic_'+comp_str+'_setstr'+str(set_str)+'_cvdrepeatedevents_final.pickle','rb') as f:
        TUPLE1, TUPLE2, TUPLE3 = pickle.load(f)
    #sfile = bz2.BZ2File('mimic_'+comp_str+'_setstr'+str(set_str)+'_cvdrepeatedevents_final.pickle','rb')
    #TUPLE1, TUPLE2, TUPLE3 = pickle.load(sfile, encoding = 'latin1')        
    (x_dim, x_dim_cont, x_dim_bin, data, time, label, mask1, mask2, mask3, data_mi, target) = TUPLE3
    
with open('./data/mimic_'+comp_str+'_cvdrepeatedevents_label.pickle','rb') as f:
    pbc2_label, train_nric1, valid_nric1, test_nric1, train_nric2, valid_nric2, test_nric2, train_nric3, valid_nric3, test_nric3, train_nric4, valid_nric4, test_nric4, train_nric5, valid_nric5, test_nric5 = pickle.load(f)

train_nric_list = locals()['train_nric'+str(set_str)]
valid_nric_list = locals()['valid_nric'+str(set_str)]
test_nric_list = locals()['test_nric'+str(set_str)]
train_indices = pbc2_label[pbc2_label['SUBJECT_ID'].isin(train_nric_list)]['id'].unique()-1    
valid_indices = pbc2_label[pbc2_label['SUBJECT_ID'].isin(valid_nric_list)]['id'].unique()-1
test_indices = pbc2_label[pbc2_label['SUBJECT_ID'].isin(test_nric_list)]['id'].unique()-1

pred_time = [122] # prediction time (in months)
eval_time = [12, 24, 60] # months evaluation time (for C-index and Brier-Score)


_, num_Event, num_Category  = np.shape(mask1)  # dim of mask3: [subj, Num_Event, Num_Category]
max_length                  = np.shape(data)[1]


file_path = '{}'.format(data_mode)

if not os.path.exists(file_path):
    os.makedirs(file_path)


# ### 2. Set Hyper-Parameters
# ##### - Play with your own hyper-parameters!

# In[ ]:


burn_in_mode                = 'OFF' #{'ON', 'OFF'}
boost_mode                  = 'OFF' #{'ON', 'OFF'}

##### HYPER-PARAMETERS
new_parser = {'mb_size': 32,

             'iteration_burn_in': 3000,
             'iteration': 35000,

             'keep_prob': 0.6,
             'lr_train': 1e-4,

             'h_dim_RNN': 100,
             'h_dim_FC' : 2,
             'num_layers_RNN':2,
             'num_layers_ATT':2,
             'num_layers_CS' :2,

             'RNN_type':'LSTM', #{'LSTM', 'GRU'}

             'FC_active_fn' : tf.nn.tanh,#tf.nn.tanh,#None,#tf.nn.relu,
             'RNN_active_fn': tf.nn.relu, #tf.nn.tanh

            'reg_W'         : 1e-5,
            'reg_W_out'     : 0.,

             'alpha' :1.0,
             'beta'  :1.0,
             'gamma' :1.0,
             'delta' :0.19,
             'time_scale':150
}


# INPUT DIMENSIONS
input_dims                  = { 'x_dim'         : x_dim,
                                'x_dim_cont'    : x_dim_cont,
                                'x_dim_bin'     : x_dim_bin,
                                'num_Event'     : num_Event,
                                'num_Category'  : num_Category,
                                'max_length'    : max_length }

# NETWORK HYPER-PARMETERS
network_settings            = { 'h_dim_RNN'         : new_parser['h_dim_RNN'],
                                'h_dim_FC'          : new_parser['h_dim_FC'],
                                'num_layers_RNN'    : new_parser['num_layers_RNN'],
                                'num_layers_ATT'    : new_parser['num_layers_ATT'],
                                'num_layers_CS'     : new_parser['num_layers_CS'],
                                'RNN_type'          : new_parser['RNN_type'],
                                'FC_active_fn'      : new_parser['FC_active_fn'],
                                'RNN_active_fn'     : new_parser['RNN_active_fn'],
                                'initial_W'         : tf.contrib.layers.xavier_initializer(),

                                'reg_W'             : new_parser['reg_W'],
                                'reg_W_out'         : new_parser['reg_W_out']
                                 }


mb_size           = new_parser['mb_size']
iteration         = new_parser['iteration']
iteration_burn_in = new_parser['iteration_burn_in']

keep_prob         = new_parser['keep_prob']
lr_train          = new_parser['lr_train']

alpha             = new_parser['alpha']
beta              = new_parser['beta']
gamma             = new_parser['gamma']
delta             = new_parser['delta']

# SAVE HYPERPARAMETERS
log_name = file_path + '/hyperparameters_log.txt'
save_logging(new_parser, log_name)


# ### 3. Split Dataset into Train/Valid/Test Sets

# In[ ]:


### TRAINING-TESTING SPLIT
if fload == 0:

    tr_data, va_data, te_data = data[train_indices], data[valid_indices], data[test_indices]
    tr_data_mi, va_data_mi, te_data_mi = data_mi[train_indices], data_mi[valid_indices], data_mi[test_indices]
    tr_time, va_time, te_time = time[train_indices], time[valid_indices], time[test_indices]
    tr_label, va_label, te_label = label[train_indices], label[valid_indices], label[test_indices]
    tr_mask1, va_mask1, te_mask1 = mask1[train_indices], mask1[valid_indices], mask1[test_indices]
    tr_mask2, va_mask2, te_mask2 = mask2[train_indices], mask2[valid_indices], mask2[test_indices]
    tr_mask3, va_mask3, te_mask3 = mask3[train_indices], mask3[valid_indices], mask3[test_indices]
    tr_tgt, va_tgt, te_tgt = target[train_indices], target[valid_indices], target[test_indices]
    
    if boost_mode == 'ON':
        tr_data, tr_data_mi, tr_time, tr_label, tr_mask1, tr_mask2, tr_mask3 = f_get_boosted_trainset(tr_data, tr_data_mi, tr_time, tr_label, tr_mask1, tr_mask2, tr_mask3)


    TUPLE1 = (tr_data,te_data, tr_data_mi, te_data_mi, tr_time,te_time, tr_label,te_label, tr_mask1,te_mask1, tr_mask2,te_mask2, tr_mask3,te_mask3, tr_tgt, te_tgt)
    TUPLE2 = (tr_data,va_data, tr_data_mi, va_data_mi, tr_time,va_time, tr_label,va_label, tr_mask1,va_mask1, tr_mask2,va_mask2, tr_mask3,va_mask3, tr_tgt, va_tgt)
    with open('mimic_'+comp_str+'_setstr'+str(set_str)+'_cvdrepeatedevents_final.pickle','wb') as f:
        pickle.dump([TUPLE1, TUPLE2, TUPLE3],f)
    #sfile = bz2.BZ2File('mimic_'+comp_str+'_setstr'+str(set_str)+'_cvdrepeatedevents_final.pickle','wb')
    #pickle.dump([TUPLE1, TUPLE2, TUPLE3],sfile)
else:
    (tr_data,te_data, tr_data_mi, te_data_mi, tr_time,te_time, tr_label,te_label, tr_mask1,te_mask1, tr_mask2,te_mask2, tr_mask3,te_mask3, tr_tgt, te_tgt) = TUPLE1
    (tr_data,va_data, tr_data_mi, va_data_mi, tr_time,va_time, tr_label,va_label, tr_mask1,va_mask1, tr_mask2,va_mask2, tr_mask3,va_mask3, tr_tgt, va_tgt) = TUPLE2
    
# ### 4. Train the Network

# In[ ]:


##### CREATE DYNAMIC-DEEPFHT NETWORK
#from class_DeepLongitudinal_rmtpp_edited_noattention_allsequences import Model_Longitudinal_Attention
tf.reset_default_graph()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

model = Model_Longitudinal_Attention(sess, "Dyanmic-DeepHit", input_dims, network_settings)
saver = tf.train.Saver()

sess.run(tf.global_variables_initializer())

 
### TRAINING - BURN-IN
if burn_in_mode == 'ON':
    print( "BURN-IN TRAINING ...")
    for itr in range(iteration_burn_in):
        x_mb, x_mi_mb, k_mb, t_mb, m1_mb, m2_mb, m3_mb, tgt_mb = f_get_minibatch(mb_size, tr_data, tr_data_mi, tr_label, tr_time, tr_mask1, tr_mask2, tr_mask3, tr_tgt)
        DATA = (x_mb, k_mb, t_mb)
        MISSING = (x_mi_mb)

        _, loss_curr = model.train_burn_in(DATA, MISSING, keep_prob, lr_train,tgt_mb/new_parser['time_scale'])

        if (itr+1)%1000 == 0:
            print('itr: {:04d} | loss: {:.4f}'.format(itr+1, loss_curr))


### TRAINING - MAIN
print( "MAIN TRAINING ...")
min_valid = 0

for itr in range(iteration):
    x_mb, x_mi_mb, k_mb, t_mb, m1_mb, m2_mb, m3_mb, tgt_mb = f_get_minibatch(mb_size, tr_data, tr_data_mi, tr_label, tr_time, tr_mask1, tr_mask2, tr_mask3, tr_tgt)
    DATA = (x_mb, k_mb, t_mb)
    MASK = (m1_mb, m2_mb, m3_mb)
    MISSING = (x_mi_mb)
    PARAMETERS = (alpha, beta, gamma, delta)
    tgt_mb = tgt_mb/new_parser['time_scale'] # scaling by 12 months

    _, loss_curr = model.train(DATA, MASK, MISSING, PARAMETERS, keep_prob, lr_train,tgt_mb)
    #train_out, train_loss1, train_loss2 = model.eval_dcal(x_mb, x_mi_mb, tgt_mb, k_mb, delta)
    #valid_out, valid_loss1, valid_loss2 = model.eval_dcal(va_data, va_data_mi, va_tgt/new_parser['time_scale'], va_label, delta)
    
    if (itr+1)%1000 == 0:
        print('itr: {:04d} | train_loss1: {:.4f}'.format(itr+1, loss_curr))
        #print('itr: {:04d} | train_loss1: {:.4f}'.format(itr+1, train_loss1))
        #print('itr: {:04d} | train_loss2: {:.4f}'.format(itr+1, train_loss2))
        #print('itr: {:04d} | train_loss_total: {:.4f}'.format(itr+1, loss_curr))        
        #print('itr: {:04d} | valid_loss1: {:.4f}'.format(itr+1, valid_loss1))
        #print('itr: {:04d} | valid_loss2: {:.4f}'.format(itr+1, valid_loss2))        

    ### VALIDATION  (based on average C-index of our interest)
    if (itr+1)%1000 == 0:        
        risk_all = f_get_risk_predictions(sess, model, va_data, va_data_mi, pred_time, eval_time, new_parser)
        
        for p, p_time in enumerate(pred_time):
            val_result1 = np.zeros([num_Event, len(eval_time)])
            
            for t, t_time in enumerate(eval_time):                
                for k in range(num_Event):
                    val_result1[k, t] = c_index(risk_all[k][:, p, t], va_tgt, (va_label[:,0] == k+1).astype(int), int(t_time)) #-1 for no event (not comparable)
            
            if p == 0:
                val_final1 = val_result1
            else:
                val_final1 = np.append(val_final1, val_result1, axis=0)

        tmp_valid = np.mean(val_final1)
        print('tmp_valid : ' + str(tmp_valid))

        if tmp_valid >  min_valid:
            min_valid = tmp_valid
            print(val_final1)
            saver.save(sess, file_path + '/model')
            print( 'updated.... average c-index = ' + str('%.4f' %(tmp_valid)))


# ### 5. Test the Trained Network

# In[ ]:


saver.restore(sess, file_path + '/model')

risk_all = f_get_risk_predictions(sess, model, te_data, te_data_mi, pred_time, eval_time, new_parser)

for p, p_time in enumerate(pred_time):
    result1, result2 = np.zeros([num_Event, len(eval_time)]), np.zeros([num_Event, len(eval_time)])

    for t, t_time in enumerate(eval_time):                
        for k in range(num_Event):
            result1[k, t] = c_index(risk_all[k][:, p, t], te_tgt, (te_label[:,0] == k+1).astype(int), int(t_time)) #-1 for no event (not comparable)
            result2[k, t] = brier_score(risk_all[k][:, p, t], te_tgt, (te_label[:,0] == k+1).astype(int), int(t_time)) #-1 for no event (not comparable)
    
    if p == 0:
        final1, final2 = result1, result2
    else:
        final1, final2 = np.append(final1, result1, axis=0), np.append(final2, result2, axis=0)
        
        
row_header = []
for p_time in pred_time:
    for t in range(num_Event):
        row_header.append('pred_time {}: event_{}'.format(p_time,k+1))
            
col_header = []
for t_time in eval_time:
    col_header.append('eval_time {}'.format(t_time))

# c-index result
df1 = pd.DataFrame(final1, index = row_header, columns=col_header)

# brier-score result
df2 = pd.DataFrame(final2, index = row_header, columns=col_header)

### PRINT RESULTS
print('========================================================')
print('--------------------------------------------------------')
print('- C-INDEX: ')
print(df1)
print('--------------------------------------------------------')
print('- BRIER-SCORE: ')
print(df2)
print('========================================================')

aa, rmse_overall_v1 = pred_next_eventtime(sess, model, te_data, te_data_mi, te_label, te_tgt, pred_time, new_parser, 100)
# te_pos_indices = np.where(te_label == 1)[0]
# te_predeventtime = np.squeeze(aa[0][te_pos_indices])
# te_realeventtime = np.squeeze(te_tgt[te_pos_indices])

# yr1_indices = np.where(te_realeventtime<13)[0]
# yr2_indices = np.where((te_realeventtime>12) & (te_realeventtime<25))[0]
# yr3_indices = np.where((te_realeventtime>24) & (te_realeventtime<37))[0]
# yr4_indices = np.where((te_realeventtime>36) & (te_realeventtime<49))[0]
# yr5_indices = np.where(te_realeventtime < 61)[0]
# mt5yr_indices = np.where(te_realeventtime > 60)[0]

# yr1_rmse = np.sqrt(mean_squared_error(te_realeventtime[yr1_indices],te_predeventtime[yr1_indices]))
# yr2_rmse = np.sqrt(mean_squared_error(te_realeventtime[yr2_indices],te_predeventtime[yr2_indices]))
# yr3_rmse = np.sqrt(mean_squared_error(te_realeventtime[yr3_indices],te_predeventtime[yr3_indices]))
# yr4_rmse = np.sqrt(mean_squared_error(te_realeventtime[yr4_indices],te_predeventtime[yr4_indices]))
# yr5_rmse = np.sqrt(mean_squared_error(te_realeventtime[yr5_indices],te_predeventtime[yr5_indices]))
# mtyr5_rmse = np.sqrt(mean_squared_error(te_realeventtime[mt5yr_indices],te_predeventtime[mt5yr_indices]))

# np.array([yr1_rmse, yr2_rmse, yr3_rmse, yr4_rmse, yr5_rmse, mtyr5_rmse])


# ===== Want to only test the c-index of samples with repeated events ===== #
repeated_data_test = pd.read_csv('/home/minmin/StayHome/Dynamic-DeepHit-master/data/Intermediate/folder'+str(set_str)+'/test_day.csv')
repeated_data_train = pd.read_csv('/home/minmin/StayHome/Dynamic-DeepHit-master/data/Intermediate/folder'+str(set_str)+'/train_day.csv')
repeated_data_valid = pd.read_csv('/home/minmin/StayHome/Dynamic-DeepHit-master/data/Intermediate/folder'+str(set_str)+'/valid_day.csv')
repeated_test_nric = set(list(repeated_data_test['id']) + list(repeated_data_train['id']) + list(repeated_data_valid['id']))
repeated_test_indices = pbc2_label[pbc2_label['SUBJECT_ID'].isin(repeated_test_nric)]['id'].unique()-1
re_te_data = data[repeated_test_indices]
re_te_data_mi = data_mi[repeated_test_indices]
re_te_time = time[repeated_test_indices]
re_te_label = label[repeated_test_indices]
re_te_mask1 = mask1[repeated_test_indices]
re_te_mask2 = mask2[repeated_test_indices]
re_te_mask3 = mask3[repeated_test_indices]
re_te_tgt = target[repeated_test_indices]

# risk_all = f_get_risk_predictions(sess, model, re_te_data, re_te_data_mi, pred_time, eval_time, new_parser)

# for p, p_time in enumerate(pred_time):
#     result1, result2 = np.zeros([num_Event, len(eval_time)]), np.zeros([num_Event, len(eval_time)])

#     for t, t_time in enumerate(eval_time):                
#         for k in range(num_Event):
#             result1[k, t] = c_index(risk_all[k][:, p, t], re_te_tgt, (re_te_label[:,0] == k+1).astype(int), int(t_time)) #-1 for no event (not comparable)
#             result2[k, t] = brier_score(risk_all[k][:, p, t], re_te_tgt, (re_te_label[:,0] == k+1).astype(int), int(t_time)) #-1 for no event (not comparable)
    
#     if p == 0:
#         final1, final2 = result1, result2
#     else:
#         final1, final2 = np.append(final1, result1, axis=0), np.append(final2, result2, axis=0)
        
        
# row_header = []
# for p_time in pred_time:
#     for t in range(num_Event):
#         row_header.append('pred_time {}: event_{}'.format(p_time,k+1))
            
# col_header = []
# for t_time in eval_time:
#     col_header.append('eval_time {}'.format(t_time))

# # c-index result
# df1 = pd.DataFrame(final1, index = row_header, columns=col_header)

# # brier-score result
# df2 = pd.DataFrame(final2, index = row_header, columns=col_header)

# ### PRINT RESULTS
# print('========================================================')
# print('--------------------------------------------------------')
# print('- C-INDEX: ')
# print(df1)
# print('--------------------------------------------------------')
# print('- BRIER-SCORE: ')
# print(df2)
# print('========================================================')

aa, rmse_overall_repeated = pred_next_eventtime(sess, model, re_te_data, re_te_data_mi, re_te_label, re_te_tgt, pred_time, new_parser, 100)

# event_observed_1yr = np.zeros(len(re_te_label))
# event_observed_1yr[np.where(re_te_time < 12)[0]] = 1
# event_observed_2yr = np.zeros(len(re_te_label))
# event_observed_2yr[np.where(re_te_time < 24)[0]] = 1
# event_observed_5yr = np.zeros(len(re_te_label))
# event_observed_5yr[np.where(re_te_time < 60)[0]] = 1
# c_ind_1yr = concordance_index(re_te_tgt,np.squeeze(risk_all[0])[:,0], event_observed_1yr)
# c_ind_2yr = concordance_index(re_te_tgt,np.squeeze(risk_all[0])[:,1], event_observed_2yr)
# c_ind_5yr = concordance_index(re_te_tgt,np.squeeze(risk_all[0])[:,2], event_observed_5yr)
# re_brier_1 = brier_score(np.squeeze(risk_all[0])[:,0], re_te_tgt, (re_te_label[:,0] == 1).astype(int), 12)
# re_brier_2 = brier_score(np.squeeze(risk_all[0])[:,1], re_te_tgt, (re_te_label[:,0] == 1).astype(int), 24)
# re_brier_5 = brier_score(np.squeeze(risk_all[0])[:,2], re_te_tgt, (re_te_label[:,0] == 1).astype(int), 60)

# re_te_pos_indices = np.where(re_te_label == 1)[0]
# re_te_predeventtime = np.squeeze(aa[0][re_te_pos_indices])
# re_te_realeventtime = np.squeeze(re_te_tgt[re_te_pos_indices])

# yr1_indices = np.where(re_te_realeventtime<13)[0]
# yr2_indices = np.where((re_te_realeventtime>12) & (re_te_realeventtime<25))[0]
# yr3_indices = np.where((re_te_realeventtime>24) & (re_te_realeventtime<37))[0]
# yr4_indices = np.where((re_te_realeventtime>36) & (re_te_realeventtime<49))[0]
# yr5_indices = np.where(re_te_realeventtime < 61)[0]
# mt5yr_indices = np.where(re_te_realeventtime > 60)[0]

# yr1_rmse = np.sqrt(mean_squared_error(re_te_realeventtime[yr1_indices],re_te_predeventtime[yr1_indices]))
# yr2_rmse = np.sqrt(mean_squared_error(re_te_realeventtime[yr2_indices],re_te_predeventtime[yr2_indices]))
# yr3_rmse = np.sqrt(mean_squared_error(re_te_realeventtime[yr3_indices],re_te_predeventtime[yr3_indices]))
# yr4_rmse = np.sqrt(mean_squared_error(re_te_realeventtime[yr4_indices],re_te_predeventtime[yr4_indices]))
# yr5_rmse = np.sqrt(mean_squared_error(re_te_realeventtime[yr5_indices],re_te_predeventtime[yr5_indices]))
# mtyr5_rmse = np.sqrt(mean_squared_error(re_te_realeventtime[mt5yr_indices],re_te_predeventtime[mt5yr_indices]))

# np.array([yr1_rmse, yr2_rmse, yr3_rmse, yr4_rmse, yr5_rmse, mtyr5_rmse])

curr_data = te_data#va_data#x_mb
curr_data_mi = te_data_mi#va_data_mi#x_mi_mb
curr_tgt = te_tgt/new_parser['time_scale']#va_tgt/150#tgt_mb
curr_label = te_label#va_label#k_mb
pred = _f_get_pred(sess, model, curr_data,  curr_data_mi, 122)
delta_t = curr_tgt
risk = np.exp((1/0.19)*np.exp(pred) - (1/0.19)*np.exp(pred+0.19*delta_t))
risk = 1-risk
bins = np.array([[0, 0.1], [0.1, 0.2], [0.2, 0.3], [0.3, 0.4], [0.4, 0.5], [0.5, 0.6], [0.6, 0.7], [0.7, 0.8], [0.8, 0.9],[0.9, 1]])
crf_mat = np.tile(np.transpose(risk),[10,1])
map1 = np.concatenate((crf_mat,bins),axis=1)

bb = np.tile(bins[:,1:2],[1,curr_data.shape[0]])
aa = np.tile(bins[:,0:1],[1,curr_data.shape[0]])
comp1 = -1000*(crf_mat - aa)*(bb-crf_mat)
comp2 = -1000*(aa-crf_mat)
common1 = np.ones_like(crf_mat)#1/(1+tf.exp(comp1))
common2 = (bb-crf_mat)/(1-crf_mat)#(1/(1+tf.exp(comp1))) * (b-crf_in)/(1-crf_in)
common3 = (bb-aa)/(1-crf_mat)#(1/(1+tf.exp(comp2)))*(b-a)/(1-crf_in)


boolean_array1 = np.equal(np.less(crf_mat,bb),np.greater_equal(crf_mat,aa))
boolean_array1 = boolean_array1.astype(int)
boolean_array2 = np.less(crf_mat,aa)
self_k = np.tile(np.transpose(curr_label),[10,1])

dcal_uncen = common1 * self_k * boolean_array1
dcal_cen = common2 * (1 - self_k) * boolean_array1 + common3 * (1-self_k) * boolean_array2
dcal = dcal_uncen + dcal_cen
dcal = np.mean(dcal,axis=1)
dcal_final = sum((dcal - 0.1) ** 2)
dcal_final = np.sqrt(dcal_final)

def quad_func(t,c):
    return c*t*np.exp(new_parser['delta']*t-(c/new_parser['delta'])*(np.exp(new_parser['delta']*t)-1))

pred = _f_get_pred(sess, model, te_data, te_data_mi, pred_time[0])
preds_i = []
C = np.exp(pred).reshape(-1)
for c_ in C:
    val, _err = quad(quad_func, 0, np.inf,args = (c_,))
    preds_i.append(val)
preds_i = np.array(preds_i) * new_parser['time_scale']

retained_indices = np.where(te_label == 1)[0]
predicted_next_event_time = preds_i * np.squeeze(te_label)
true_event_time = np.squeeze(te_tgt) * np.squeeze(te_label)
rmse_overall_v2 = np.sqrt(mean_squared_error(predicted_next_event_time[retained_indices], true_event_time[retained_indices]))

