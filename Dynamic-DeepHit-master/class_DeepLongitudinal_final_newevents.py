import numpy as np
import tensorflow as tf
import random

from tensorflow.contrib.layers import fully_connected as FC_Net
from tensorflow.python.ops.rnn import _transpose_batch_time


import utils_network as utils

_EPSILON = 1e-08
MAX_VALUE = 88



##### USER-DEFINED FUNCTIONS
def log(x):
    return tf.log(x + _EPSILON)

def div(x, y):
    return tf.div(x, (y + _EPSILON))

def tf_exp(x):
    return tf.exp(tf.minimum(x,MAX_VALUE))
    
def get_seq_length(sequence):
    used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
    tmp_length = tf.reduce_sum(used, 1)
    tmp_length = tf.cast(tmp_length, tf.int32)
    return tmp_length


class Model_Longitudinal_Attention:
    # def __init__(self, sess, name, mb_size, input_dims, network_settings):
    def __init__(self, sess, name, input_dims, network_settings):
        self.sess               = sess
        self.name               = name

        # INPUT DIMENSIONS
        self.x_dim              = input_dims['x_dim']
        self.x_dim_cont         = input_dims['x_dim_cont']
        self.x_dim_bin          = input_dims['x_dim_bin']

        self.num_Event          = input_dims['num_Event']
        self.num_Category       = input_dims['num_Category']
        self.max_length         = input_dims['max_length']

        # NETWORK HYPER-PARMETERS
        self.h_dim1             = network_settings['h_dim_RNN']
        self.h_dim2             = network_settings['h_dim_FC']
        self.num_layers_RNN     = network_settings['num_layers_RNN']
        self.num_layers_ATT     = network_settings['num_layers_ATT']
        self.num_layers_CS      = network_settings['num_layers_CS']

        self.RNN_type           = network_settings['RNN_type']

        self.FC_active_fn       = network_settings['FC_active_fn']
        self.RNN_active_fn      = network_settings['RNN_active_fn']
        self.initial_W          = network_settings['initial_W']
        
        self.reg_W              = tf.contrib.layers.l1_regularizer(scale=network_settings['reg_W'])
        self.reg_W_out          = tf.contrib.layers.l1_regularizer(scale=network_settings['reg_W_out'])

        self._build_net()


    def _build_net(self):
        with tf.variable_scope(self.name):
            #### PLACEHOLDER DECLARATION
            self.mb_size     = tf.placeholder(tf.int32, [], name='batch_size')

            self.lr_rate     = tf.placeholder(tf.float32)
            self.keep_prob   = tf.placeholder(tf.float32)                                                      #keeping rate
            self.a           = tf.placeholder(tf.float32)
            self.b           = tf.placeholder(tf.float32)
            self.c           = tf.placeholder(tf.float32)
            self.d           = tf.placeholder(tf.float32)

            self.x           = tf.placeholder(tf.float32, shape=[None, self.max_length, self.x_dim])
            self.x_mi        = tf.placeholder(tf.float32, shape=[None, self.max_length, self.x_dim])           #this is the missing indicator (including for cont. & binary) (includes delta)
            self.k           = tf.placeholder(tf.float32, shape=[None, 1])                                     #event/censoring label (censoring:0)
            self.t           = tf.placeholder(tf.float32, shape=[None, 1])
            self.tgt         = tf.placeholder(tf.float32, shape=[None, 1])


            self.fc_mask1    = tf.placeholder(tf.float32, shape=[None, self.num_Event, self.num_Category])     #for denominator
            self.fc_mask2    = tf.placeholder(tf.float32, shape=[None, self.num_Event, self.num_Category])     #for Loss 1
            self.fc_mask3    = tf.placeholder(tf.float32, shape=[None, self.num_Category])                     #for Loss 2

            
            seq_length     = get_seq_length(self.x)
            tmp_range      = tf.expand_dims(tf.range(0, self.max_length, 1), axis=0)
            max_length_vec = tf.ones_like(seq_length) * self.max_length
            
            self.rnn_mask1 = tf.cast(tf.less_equal(tmp_range, tf.expand_dims(seq_length - 1, axis=1)), tf.float32)            
            self.rnn_mask2 = tf.cast(tf.equal(tmp_range, tf.expand_dims(seq_length - 1, axis=1)), tf.float32) 
            self.att_mask = tf.cast(tf.equal(tmp_range, tf.expand_dims(max_length_vec - 1, axis = 1)), tf.float32)
            
            
            ### DEFINE LOOP FUNCTION FOR RAW_RNN w/ TEMPORAL ATTENTION
            def loop_fn_att(time, cell_output, cell_state, loop_state):

                emit_output = cell_output 

                if cell_output is None:  # time == 0
                    next_cell_state = cell.zero_state(self.mb_size, tf.float32)
                    next_loop_state = loop_state_ta
                else:
                    next_cell_state = cell_state
                    tmp_h = utils.create_concat_state(next_cell_state, self.num_layers_RNN, self.RNN_type)

                    e = utils.create_FCNet(tmp_h, self.num_layers_ATT, self.h_dim2, 
                                           tf.nn.tanh, 1, None, self.initial_W, keep_prob=self.keep_prob)
                    e = tf_exp(e)

                    next_loop_state = (loop_state[0].write(time-1, e),                # save att power (e_{j})
                                       loop_state[1].write(time-1, tmp_h))  # save all the hidden states

                # elements_finished = (time >= seq_length)
                elements_finished = (time >= self.max_length)

                #this gives the break-point (no more recurrence after the max_length)
                finished = tf.reduce_all(elements_finished)    
                next_input = tf.cond(finished, lambda: tf.zeros([self.mb_size, 2*self.x_dim], dtype=tf.float32),  # [x_hist, mi_hist]
                                               lambda: inputs_ta.read(time))

                return (elements_finished, next_input, next_cell_state, emit_output, next_loop_state)


            all_x = tf.concat([self.x, self.x_mi], axis=2)

            #extract inputs for the temporal attention: mask (to incorporate only the measured time) and x_{M}
            seq_length     = get_seq_length(all_x)
            rnn_mask_att   = tf.cast(tf.not_equal(tf.reduce_sum(all_x, reduction_indices=2), 0), dtype=tf.float32)  #[mb_size, max_length-1], 1:measurements 0:no measurements
            

            ##### SHARED SUBNETWORK: RNN w/ TEMPORAL ATTENTION
            #change the input tensor to TensorArray format with [max_length, mb_size, x_dim]
            inputs_ta = tf.TensorArray(dtype=tf.float32, size=self.max_length).unstack(_transpose_batch_time(all_x), name = 'Shared_Input')


            #create a cell with RNN hyper-parameters (RNN types, #layers, #nodes, activation functions, keep proability)
            cell = utils.create_rnn_cell(self.h_dim1, self.num_layers_RNN, self.keep_prob, 
                                         self.RNN_type, self.RNN_active_fn)

            #define the loop_state TensorArray for information from rnn time steps
            loop_state_ta = (tf.TensorArray(size=self.max_length, dtype=tf.float32),  #e values (e_{j})
                             tf.TensorArray(size=self.max_length, dtype=tf.float32))  #hidden states (h_{j})
            
            rnn_outputs_ta, self.rnn_final_state, loop_state_ta = tf.nn.raw_rnn(cell, loop_fn_att)
            #rnn_outputs_ta  : TensorArray
            #rnn_final_state : Tensor
            #rnn_states_ta   : (TensorArray, TensorArray)

            rnn_outputs = _transpose_batch_time(rnn_outputs_ta.stack())
            # rnn_outputs =  tf.reshape(rnn_outputs, [-1, self.max_length-1, self.h_dim1])

            rnn_states  = _transpose_batch_time(loop_state_ta[1].stack())
            self.rnn_states = rnn_states
            self.rnn_states_last = tf.reduce_sum(tf.slice(rnn_states,[0,self.max_length-1,0],[self.mb_size, 1,-1]),axis=1)

            att_weight  = _transpose_batch_time(loop_state_ta[0].stack()) #e_{j}
            att_weight  = tf.reshape(att_weight, [-1, self.max_length]) * rnn_mask_att # masking to set 0 for the unmeasured e_{j}

            #get a_{j} = e_{j}/sum_{l=1}^{M-1}e_{l}
            self.att_weight  = div(att_weight,(tf.reduce_sum(att_weight, axis=1, keepdims=True) + _EPSILON)) #softmax (tf.exp is done, previously)

            # 1) expand att_weight to hidden state dimension, 2) c = \sum_{j=1}^{M} a_{j} x h_{j}
            self.context_vec = tf.reduce_sum(tf.tile(tf.reshape(self.att_mask, [-1, self.max_length, 1]), [1, 1, self.num_layers_RNN*self.h_dim1]) * rnn_states, axis=1) # Using the last hidden vector of max_length for computation
            #self.context_vec = tf.reduce_sum(tf.tile(tf.reshape(self.rnn_mask2, [-1, self.max_length, 1]), [1, 1, self.num_layers_RNN*self.h_dim1]) * rnn_states, axis=1) # Using the last hidden vector for every sequence for computation
            #self.context_vec = tf.reduce_sum(tf.tile(tf.reshape(self.att_weight, [-1, self.max_length, 1]), [1, 1, self.num_layers_RNN*self.h_dim1]) * rnn_states, axis=1) # Using context vector for computation

            ##### CS-SPECIFIC SUBNETWORK w/ FCNETS 
            #inputs = self.rnn_states_last
            inputs = self.context_vec
            #inputs = tf.concat([x_last, rnn_outputs[:,-1,:]], axis=1)


            #1 layer for combining inputs
            h = FC_Net(inputs, self.h_dim2, activation_fn=self.FC_active_fn, weights_initializer=self.initial_W, scope="Layer1")
            h = tf.nn.dropout(h, keep_prob=self.keep_prob)

            # (num_layers_CS-1) layers for cause-specific (num_Event subNets)
            out = []
            for _ in range(self.num_Event):
                cs_out = utils.create_FCNet(h, (self.num_layers_CS), self.h_dim2, self.FC_active_fn, self.h_dim2, self.FC_active_fn, self.initial_W, self.reg_W, self.keep_prob)
                out.append(cs_out)
            out = tf.stack(out, axis=1) # stack referenced on subject
            out = tf.reshape(out, [-1, self.num_Event*self.h_dim2])
            out = tf.nn.dropout(out, keep_prob=self.keep_prob)

            out = FC_Net(out, self.num_Event * 1, activation_fn=None, 
                         weights_initializer=self.initial_W, weights_regularizer=self.reg_W_out, scope="Output")
            out = tf.reshape(out, [-1, self.num_Event, 1])
            self.out = tf.reduce_sum(out, reduction_indices=2)

            ##### GET LOSS FUNCTIONS
            self.loss_Log_Likelihood()      #get loss1: Log-Likelihood loss
            #self.loss_dcal()      #get loss3: RNN prediction loss

            self.LOSS_TOTAL     = self.a*self.LOSS_1 + tf.losses.get_regularization_loss()

            self.solver         = tf.train.AdamOptimizer(learning_rate=self.lr_rate).minimize(self.LOSS_TOTAL)


    ### LOSS-FUNCTION 1 -- Log-likelihood loss
    def loss_Log_Likelihood(self):
        ret_term = -self.out - self.d*self.tgt
        ret_term = self.k * ret_term
        common_term = -(1/self.d)*tf_exp(self.out) + (1/self.d)*tf_exp(self.out+self.d*self.tgt)
        add_term = -self.k * log(1-tf_exp(-common_term))

        self.LOSS_1 = tf.reduce_mean(ret_term + common_term + add_term)

    ### LOSS-FUNCTION 3 -- RNN prediction loss
    def loss_dcal(self):
        bins = tf.constant([[0, 0.1], [0.1, 0.2], [0.2, 0.3], [0.3, 0.4], [0.4, 0.5], [0.5, 0.6], [0.6, 0.7], [0.7, 0.8], [0.8, 0.9],[0.9, 1]])
        bins = tf.convert_to_tensor(bins, dtype=tf.float32)
        surv_term = (1/self.d)*tf_exp(self.out) - (1/self.d)*tf_exp(self.out+self.d*self.tgt)
        crf = 1-tf_exp(surv_term) # cumulative risk function
        crf = tf.transpose(tf.tile(crf, [1,10]))

        map1 = tf.concat([crf, bins], axis=1)
        
        def common_obj_ab(x):
            crf_in = x[0:self.mb_size]
            a = x[self.mb_size]
            b = x[self.mb_size+1]
            comp1 = -1000*(crf_in-a)*(b-crf_in)#-10*(crf_in-a)*(b-crf_in)#
            comp2 = -1000*(a-crf_in) #-10*(a-crf_in)#
            common1 = 1/(1+tf_exp(comp1)) #tf.ones_like(crf_in)#
            common2 = (1/(1+tf_exp(comp1))) * (b-crf_in)/(1-crf_in) #(b-crf_in)/(1-crf_in)#
            common3 = (1/(1+tf_exp(comp2)))*(b-a)/(1-crf_in) #(b-a)/(1-crf_in)#
            ret = tf.stack([common1, common2, common3])
            return ret
        
        def boolean_mask(x):
            crf_in = x[0:self.mb_size]
            a = x[self.mb_size]
            b = x[self.mb_size+1]
            boolean1 = tf.cast(tf.equal(tf.less(crf_in,b),tf.greater_equal(crf_in,a)),tf.float32)
            boolean2 = tf.cast(tf.less(crf_in,a),tf.float32)
            ret_bool = tf.stack([boolean1, boolean2])
            return ret_bool
        
        rate_result = tf.map_fn(common_obj_ab, map1)
        self_k = tf.tile(tf.transpose(self.k),[10,1])

        boolean_out = tf.map_fn(boolean_mask, map1)
        boolean_out = tf.stop_gradient(boolean_out)
        boolean_array1 = tf.squeeze(tf.slice(boolean_out,[0,0,0],[-1,1,self.mb_size]))
        boolean_array2 = tf.squeeze(tf.slice(boolean_out,[0,1,0],[-1,1,self.mb_size]))
        dcal_uncen = tf.squeeze(tf.slice(rate_result, [0,0,0],[-1,1,self.mb_size])) * self_k * boolean_array1
        dcal_uncen = tf.reduce_mean(dcal_uncen,axis=1)
        dcal_cen = tf.squeeze(tf.slice(rate_result, [0,1,0],[-1,1,self.mb_size])) * (1 - self_k) * boolean_array1 + tf.squeeze(tf.slice(rate_result, [0,2,0],[-1,1,self.mb_size])) * (1-self_k) * boolean_array2
        dcal_cen = tf.reduce_mean(dcal_cen,axis=1)
        self.LOSS_2 = tf.reduce_sum((dcal_uncen + dcal_cen - 0.1) * (dcal_uncen + dcal_cen - 0.1))   

 
    def get_cost(self, DATA, MASK, MISSING, PARAMETERS, keep_prob, lr_train):
        (x_mb, k_mb, t_mb)        = DATA
        (m1_mb, m2_mb, m3_mb)     = MASK
        (x_mi_mb)                 = MISSING
        (alpha, beta, gamma)      = PARAMETERS
        return self.sess.run(self.LOSS_TOTAL, 
                             feed_dict={self.x:x_mb, self.x_mi: x_mi_mb, self.k:k_mb, self.t:t_mb, 
                                        self.fc_mask1: m1_mb, self.fc_mask2:m2_mb, self.fc_mask3: m3_mb, 
                                        self.a:alpha, self.b:beta, self.c:gamma,
                                        self.mb_size: np.shape(x_mb)[0], self.keep_prob:keep_prob, self.lr_rate:lr_train})

    def train(self, DATA, MASK, MISSING, PARAMETERS, keep_prob, lr_train, tgt_mb):
        (x_mb, k_mb, t_mb)        = DATA
        (m1_mb, m2_mb, m3_mb)     = MASK
        (x_mi_mb)                 = MISSING
        (alpha, beta, gamma, delta)      = PARAMETERS
        return self.sess.run([self.solver, self.LOSS_TOTAL], 
                             feed_dict={self.x:x_mb, self.x_mi: x_mi_mb, self.k:k_mb, self.t:t_mb,
                                        self.fc_mask1: m1_mb, self.fc_mask2:m2_mb, self.fc_mask3: m3_mb, 
                                        self.a:alpha, self.b:beta, self.c:gamma,
                                        self.mb_size: np.shape(x_mb)[0], self.keep_prob:keep_prob, self.lr_rate:lr_train, self.tgt:tgt_mb, self.d:delta})

    def eval_dcal(self, x_test, x_mi_test, x_tgt_test, x_label_test, delta, keep_prob=1.0):
        return self.sess.run([self.out, self.LOSS_1, self.LOSS_2],
                             feed_dict={self.x: x_test, self.x_mi: x_mi_test, self.tgt: x_tgt_test,
                                        self.d: delta, self.k: x_label_test,
                                        self.mb_size: np.shape(x_test)[0], self.keep_prob: keep_prob})    
        
    def train_burn_in(self, DATA, MISSING, keep_prob, lr_train):
        (x_mb, k_mb, t_mb)        = DATA
        (x_mi_mb)                 = MISSING

        return self.sess.run([self.solver_burn_in, self.LOSS_3], 
                             feed_dict={self.x:x_mb, self.x_mi: x_mi_mb, self.k:k_mb, self.t:t_mb, 
                                        self.mb_size: np.shape(x_mb)[0], self.keep_prob:keep_prob, self.lr_rate:lr_train})
    
    def predict(self, x_test, x_mi_test, keep_prob=1.0):
        return self.sess.run(self.out, feed_dict={self.x: x_test, self.x_mi: x_mi_test, self.mb_size: np.shape(x_test)[0], self.keep_prob: keep_prob})

    def predict_z(self, x_test, x_mi_test, keep_prob=1.0):
        return self.sess.run(self.z, feed_dict={self.x: x_test, self.x_mi: x_mi_test, self.mb_size: np.shape(x_test)[0], self.keep_prob: keep_prob})

    def predict_rnnstate(self, x_test, x_mi_test, keep_prob=1.0):
        return self.sess.run(self.rnn_final_state, feed_dict={self.x: x_test, self.x_mi: x_mi_test, self.mb_size: np.shape(x_test)[0], self.keep_prob: keep_prob})

    def predict_att(self, x_test, x_mi_test, keep_prob=1.0):
        return self.sess.run(self.att_weight, feed_dict={self.x: x_test, self.x_mi: x_mi_test, self.mb_size: np.shape(x_test)[0], self.keep_prob: keep_prob})

    def predict_context_vec(self, x_test, x_mi_test, keep_prob=1.0):
        return self.sess.run(self.context_vec, feed_dict={self.x: x_test, self.x_mi: x_mi_test, self.mb_size: np.shape(x_test)[0], self.keep_prob: keep_prob})

    def get_z_mean_and_std(self, x_test, x_mi_test, keep_prob=1.0):
        return self.sess.run([self.z_mean, self.z_std], feed_dict={self.x: x_test, self.x_mi: x_mi_test, self.mb_size: np.shape(x_test)[0], self.keep_prob: keep_prob})
    
    def predict_h_inputs(self, x_test, x_mi_test, keep_prob=1.0):
        return self.sess.run([self.h, self.check_inputs], feed_dict={self.x: x_test, self.x_mi: x_mi_test, self.mb_size: np.shape(x_test)[0], self.keep_prob: keep_prob})
    
    def check(self, DATA, MASK, MISSING, PARAMETERS, keep_prob, lr_train, tgt_mb):
        (x_mb, k_mb, t_mb)        = DATA
        (m1_mb, m2_mb, m3_mb)     = MASK
        (x_mi_mb)                 = MISSING
        (alpha, beta, gamma, delta)      = PARAMETERS
        return self.sess.run([self.solver, self.LOSS_TOTAL, self.att_mask, self.context_vec, self.rnn_states_last, self.att_weight, self.rnn_states, self.rnn_mask2], 
                             feed_dict={self.x:x_mb, self.x_mi: x_mi_mb, self.k:k_mb, self.t:t_mb,
                                        self.fc_mask1: m1_mb, self.fc_mask2:m2_mb, self.fc_mask3: m3_mb, 
                                        self.a:alpha, self.b:beta, self.c:gamma,
                                        self.mb_size: np.shape(x_mb)[0], self.keep_prob:keep_prob, self.lr_rate:lr_train, self.tgt:tgt_mb, self.d:delta})