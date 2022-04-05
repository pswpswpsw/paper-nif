import numpy as np
from matplotlib import pyplot as plt
import tensorflow.compat.v1 as tf
import sys


FILENAME = './DATA/sst_train_n.npz'

data=np.load(FILENAME)['data']
mean = np.load(FILENAME)['mean']
std = np.load(FILENAME)['std']
data_raw = data 

arr=np.diff(data_raw[:,0])>0
index_max_list=np.where(arr==1)[0]


N_SENSOR = int(sys.argv[1])


if N_SENSOR == 5:
    MODEL_NAME = 'SST_NIF_SIREN_NSNAP_832_NSX_60_LSX_2_NST_60_LST_2_NP_5_NSENSOR_5'
elif N_SENSOR == 6:
    MODEL_NAME = 'SST_NIF_SIREN_NSNAP_832_NSX_60_LSX_2_NST_60_LST_2_NP_6_NSENSOR_6'
elif N_SENSOR == 7:
    MODEL_NAME = 'SST_NIF_SIREN_NSNAP_832_NSX_60_LSX_2_NST_60_LST_2_NP_7_NSENSOR_7'
elif N_SENSOR == 8:
    MODEL_NAME = 'SST_NIF_SIREN_NSNAP_832_NSX_60_LSX_2_NST_60_LST_2_NP_8_NSENSOR_8'
elif N_SENSOR == 9:
    MODEL_NAME = 'SST_NIF_SIREN_NSNAP_832_NSX_60_LSX_2_NST_60_LST_2_NP_9_NSENSOR_9'
elif N_SENSOR == 10:
    MODEL_NAME = 'SST_NIF_SIREN_NSNAP_832_NSX_60_LSX_2_NST_60_LST_2_NP_10_NSENSOR_10'
elif N_SENSOR==20:
    MODEL_NAME = 'SST_NIF_SIREN_NSNAP_832_NSX_60_LSX_2_NST_60_LST_2_NP_20_NSENSOR_20'
elif N_SENSOR==50:
    MODEL_NAME = 'SST_NIF_SIREN_NSNAP_832_NSX_60_LSX_2_NST_60_LST_2_NP_50_NSENSOR_50'
elif N_SENSOR==100:
    MODEL_NAME = 'SST_NIF_SIREN_NSNAP_832_NSX_60_LSX_2_NST_100_LST_2_NP_100_NSENSOR_100'
elif N_SENSOR==200:
    MODEL_NAME = 'SST_NIF_SIREN_NSNAP_832_NSX_60_LSX_2_NST_200_LST_2_NP_200_NSENSOR_200'
elif N_SENSOR==300:
    MODEL_NAME = 'SST_NIF_SIREN_NSNAP_832_NSX_60_LSX_2_NST_300_LST_2_NP_300_NSENSOR_300'
elif N_SENSOR==400:
    MODEL_NAME = 'SST_NIF_SIREN_NSNAP_832_NSX_60_LSX_2_NST_400_LST_2_NP_400_NSENSOR_400'
elif N_SENSOR==500:
    MODEL_NAME = 'SST_NIF_SIREN_NSNAP_832_NSX_60_LSX_2_NST_500_LST_2_NP_500_NSENSOR_500'
elif N_SENSOR==600:
    MODEL_NAME = 'SST_NIF_SIREN_NSNAP_832_NSX_60_LSX_2_NST_600_LST_2_NP_600_NSENSOR_600'
    
sensor_path = './NIF/' + str(N_SENSOR) + '_sensor_location.npz'
model_dir = './NIF/'+MODEL_NAME+'/saved_model_ckpt_150'


i_sensor = np.load(sensor_path)['i_sensor']
data_raw_ar = data_raw[:,-1].reshape(1642,-1)

# arrange sensor data
data_raw_sensor = data_raw_ar[:,i_sensor]

# load the model with graph
sess = tf.Session(graph=tf.Graph())
MODEL_LOADED=tf.saved_model.loader.load(sess, ["serve"], model_dir)
graph = sess.graph

# load input tensors
INPUT_S = graph.get_tensor_by_name('input_SENSOR:0')
INPUT_Y = graph.get_tensor_by_name('input_Y:0')
INPUT_X = graph.get_tensor_by_name('input_X:0')

# ouput tensor
OUTPUT_U = graph.get_tensor_by_name('output_u:0')

# get X,Y data for 1642 weeks
input_arr_x = data_raw[:,-3].reshape(-1,1)
input_arr_y = data_raw[:,-2].reshape(-1,1)

tmp_list = []
DIM_Y = 6317
DIM_X = 7

Num_test_weeks = 1642-832 # 104

for i in range(832+Num_test_weeks):
    print('i = ', i, ' / 832+104')
    tt = []
    for j in range(DIM_X):
        ttt = sess.run(OUTPUT_U, feed_dict={INPUT_X: input_arr_x[i*DIM_Y*DIM_X + j*DIM_Y : i*DIM_Y*DIM_X + (j+1)*DIM_Y],  
                                            INPUT_Y: input_arr_y[i*DIM_Y*DIM_X + j*DIM_Y : i*DIM_Y*DIM_X + (j+1)*DIM_Y], 
                                            INPUT_S: np.repeat(data_raw_sensor[[i],:],DIM_Y,axis=0)})
        tt.append(ttt)
    sst_pred = np.vstack(tt).ravel()
    tmp_list.append(sst_pred)
    
sst_pred = np.vstack(tmp_list)

## un-normalize the dataset
sst_pred=sst_pred*std[-1]+mean[-1]

# save the data
np.savez('./NIF/'+MODEL_NAME+'/output_arr_u.npz',output_arr_u=sst_pred)
