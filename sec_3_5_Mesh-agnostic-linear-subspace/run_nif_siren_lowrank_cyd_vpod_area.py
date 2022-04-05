from __future__ import division, print_function, absolute_import
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import argparse
import numpy as np
import time
import logging
import os
import GPUtil
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def mkdir(CASE_NAME):
    if not os.path.exists(CASE_NAME):
        os.makedirs(CASE_NAME)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
try:
    DEVICE_ID_LIST = GPUtil.getFirstAvailable(maxLoad=0.05, maxMemory=0.05)
except:
    DEVICE_ID_LIST = ["0"]
    
parser = argparse.ArgumentParser(description='config')
parser.add_argument('--TRAIN_DATA',type=str, help='normalized train data file path')
parser.add_argument('--DIM_U',type=int, help='dimension of output')
parser.add_argument('--NUM_HIDDEN_SPACE',type=int, help='width of hidden units for shape-net',default=100)
parser.add_argument('--LAYER_HIDDEN_SPACE',type=int, help='number of hidden layer for shape-net',default=3)
parser.add_argument('--NUM_HIDDEN_TIME',type=int, help='width of hidden units for para-net',default=100)
parser.add_argument('--LAYER_HIDDEN_TIME',type=int, help='number of hidden layer for para-net',default=3)
parser.add_argument('--RANK',type=int, help='rank of space-time field',default=3)
parser.add_argument('--BATCH_SIZE',type=int, help='batch size',default=6400)
parser.add_argument('--GPU_ID',type=str, help='id of GPU', default=DEVICE_ID_LIST[0])
parser.add_argument('--RUN_ID',type=str, help='id of trial run')
args = parser.parse_args()

D_PATH = args.TRAIN_DATA
DIM_U = args.DIM_U
N_SX = args.NUM_HIDDEN_SPACE
L_SX = args.LAYER_HIDDEN_SPACE
N_ST = args.NUM_HIDDEN_TIME
L_ST = args.LAYER_HIDDEN_TIME
N_P = args.RANK
BATCH_SIZE = args.BATCH_SIZE
GPU_ID = args.GPU_ID
RUN_ID = args.RUN_ID

os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)
gpu_options = tf.GPUOptions(allow_growth=True)

CASE_NAME = 'NIF_SIREN_AREA_RANK_' + str(N_P) + '_NSX_' + str(N_SX) + \
            '_LSX_'+str(L_SX) + '_NST_' + str(N_ST) + '_LST_'+str(L_ST)  + '_' + RUN_ID + '_VECPOD'
mkdir(CASE_NAME)
mkdir(CASE_NAME + '/pngs')

logging.basicConfig(filename=CASE_NAME + '/log', level=logging.INFO, format='%(message)s')
init_weight = tf.truncated_normal_initializer(stddev=1e-1)

# Training Parameters
NEPOCH = 1201
LR = 1e-5
display_epoch = 2
plot_epoch = 50 # 2000
checkpt_epoch = 100

# Shape Network Parameters
omega_0_t = 30.
omega_0_x = 30.

# paranet in-out dim
NP_I = 1
# shapenet in-out dim
NS_I = 2
NS_O = DIM_U

# total number of shape-net parameters
# NT_P = (N_P + 1)*NS_O
# NT_P = (L_SX-1)*N_SX**2 + (NS_I + NS_O + L_SX)*N_SX + NS_O

# build siren layer
class SIREN_LR(tf.keras.layers.Layer):
    def __init__(self, num_inputs, num_outputs, is_first=False, omega_0_e=30., omega_0=30., is_last=False, ns=None):
        super(SIREN_LR, self).__init__()
        self.omega_0 = omega_0
        self.omega_0_e = omega_0_e
        self.is_first = is_first
        self.num_inputs = num_inputs

        # initialize the bias
        b_init = tf.random.uniform(shape=(num_outputs,),
                                   minval=-1. / np.sqrt(self.num_inputs),
                                   maxval=1. / np.sqrt(self.num_inputs))

        if is_first:

            w_init = tf.random.uniform(shape=(num_inputs, num_outputs),
                                       minval=-1. / self.num_inputs,
                                       maxval=1. / self.num_inputs)

        elif not is_last:

            w_init = tf.random.uniform(shape=(num_inputs, num_outputs),
                                       minval=-tf.math.sqrt(6.0 / self.num_inputs) / self.omega_0,
                                       maxval=tf.math.sqrt(6.0 / self.num_inputs) / self.omega_0)

        elif ns is not None:

            # weights init: use 0.01 x HeUniform
            w_init = tf.random.uniform(shape=(num_inputs, num_outputs),
                                       minval=-np.sqrt(6.0 / num_inputs) * 1e-2,
                                       maxval=np.sqrt(6.0 / num_inputs) * 1e-2)

            tmp = np.ones((num_outputs), )
            tmp[:NS_O*ns] = tmp[:NS_O*ns] * np.sqrt(6.0 / ns) / self.omega_0_e  # last layer weights
            tmp[NS_O*ns:] = 1.0 / ns  # all biases
            scale_matrix = tf.cast(tmp, tf.float32)
            b_init = tf.random.uniform((num_outputs,), -scale_matrix, scale_matrix)

        else:
            # this corresponds to no scaling correction to SIREN
            w_init = tf.random.uniform((num_inputs, num_outputs),
                                       -tf.math.sqrt(6.0 / self.num_inputs) / self.omega_0,
                                       tf.math.sqrt(6.0 / self.num_inputs) / self.omega_0)

        self.w = tf.Variable(w_init)
        self.b = tf.Variable(b_init)

        if is_last:
            self.act = lambda x: x
        else:
            self.act = lambda x: tf.math.sin(self.omega_0 * x)

    def call(self, inputs):
        return self.act(tf.matmul(inputs, self.w) + self.b)


def NIF(t, x, y):

    ## original SIREN T
    lt1 = SIREN_LR(1, N_ST, is_first=True, omega_0=omega_0_t, omega_0_e=omega_0_x)
    lt_list = []
    for i in range(L_ST-1):
        tmpl = SIREN_LR(N_ST, N_ST, omega_0=omega_0_t, omega_0_e=omega_0_x)
        lt_list.append(tmpl)
    # l2 = SIREN(N_ST, N_P, omega_0=omega_0_t, omega_0_e=omega_0_x)
    lt3 = SIREN_LR(N_ST, N_P, omega_0=omega_0_t, omega_0_e=omega_0_x, is_last=True, ns=N_P)

    para_net = lt1(t)
    for i in range(L_ST-1):
        para_net = lt_list[i](para_net)
    # para_net = l2(para_net)
    para_net = lt3(para_net)
    para_net = tf.reshape(para_net, [-1, N_P, 1])

    # Build "shape net"
    ls1 = SIREN_LR(NS_I, N_SX, is_first=True, omega_0=omega_0_t, omega_0_e=omega_0_x)
    ls_list = []
    for i in range(L_SX-1):
        tmpl = SIREN_LR(N_SX, N_SX, omega_0=omega_0_t, omega_0_e=omega_0_x)
        ls_list.append(tmpl)
    ls3 = SIREN_LR(N_SX, N_P*DIM_U, omega_0=omega_0_t, omega_0_e=omega_0_x, is_last=True)

    xy = tf.concat((x, y), -1)
    u = ls1(xy)
    for i in range(L_SX-1):
        u = ls_list[i](u)
    phi_x = ls3(u)

    # multiplying output of paran-net and shape-net
    bias = tf.Variable(init_weight([DIM_U,]))
    u_list = []
    for i in range(DIM_U):
        tmp = tf.einsum('ai,aij->aj', phi_x[:, i*N_P:(i+1)*N_P], para_net) 
        u_list.append(tmp)

    u = tf.concat(u_list,axis=-1) + bias

    return u, para_net, phi_x

# tf Graph input
X = tf.placeholder(tf.float32, [None, 1], 'input_X')
Y = tf.placeholder(tf.float32, [None, 1], 'input_Y')
T = tf.placeholder(tf.float32, [None, 1], 'input_T')
AREA = tf.placeholder(tf.float32, [None, 1], 'input_area')
U = tf.placeholder(tf.float32, [None, DIM_U])

# Create a graph for training
output_u, para_net, phi_x = NIF(T, X, Y)


# weight the loss with area because dt is the same so we ignore
loss_op = tf.reduce_mean(tf.square(output_u - U) * AREA)

# loss_op = tf.reduce_mean(tf.square(output_u - U))

optimizer = tf.train.AdamOptimizer(LR)
grads = optimizer.compute_gradients(loss_op)
train_op = optimizer.apply_gradients(grads)

output_u = tf.identity(output_u, 'output_u')
para_net = tf.identity(para_net, 'para_net')
phi_x  = tf.identity(phi_x, 'phi_x')

# loss_op_all = tf.reduce_mean(tf.square(output_u - U))
loss_op_all = tf.reduce_mean(tf.square(output_u - U) * AREA)

init = tf.global_variables_initializer()

# load data -- AMR data of cylinder
TD = np.load(D_PATH)
TRAIN_DATA = TD['data']
mean_tr = TD['mean']
std_tr = TD['std']

# get the time index for each snapshot
arr=np.diff(TRAIN_DATA[:,0])>0
index_max_list=np.where(arr==1)[0]
i_max = index_max_list[0]
# i_max = np.where(np.diff(TRAIN_DATA[:,0])>0 == 1)[0][0]

# Start Training
# with tf.Session() as sess:
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options))

# Run the initializer
sess.run(init)
T_START = time.time()
mse_mini_batch_list = []

## print the total number of trainable parameters
total_parameters = 0
for variable in tf.trainable_variables():
    # shape is an array of tf.Dimension
    shape = variable.get_shape()
    variable_parameters = 1
    for dim in shape:
        variable_parameters *= dim.value
    total_parameters += variable_parameters
print('total trainable = ', total_parameters)


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert inputs.shape[0] == targets.shape[0]
    if shuffle:
        indices = np.arange(inputs.shape[0])
        np.random.shuffle(indices)
    for start_idx in range(0, inputs.shape[0] - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


# Keep training until reach max iterations
for epoch in range(1, NEPOCH + 1):
    for batch in iterate_minibatches(TRAIN_DATA[:,0:3], TRAIN_DATA[:,-DIM_U-1:], BATCH_SIZE, shuffle=True):
        feature_batch, batch_target = batch
        batch_t  = feature_batch[:,[0]]
        batch_x  = feature_batch[:,[1]]
        batch_y  = feature_batch[:,[2]]
        batch_u = batch_target[:,0:DIM_U]
        batch_area = batch_target[:,[DIM_U]].reshape(-1,1)

        ts = time.time()
        sess.run(train_op, {X: batch_x, T: batch_t, U: batch_u, Y:batch_y, AREA: batch_area})
        te = time.time() - ts

    if epoch % display_epoch == 0 or epoch == 1:
        # Calculate LAST BATCH loss and accuracy
        loss = sess.run(loss_op, {X: batch_x, T: batch_t, U: batch_u, Y:batch_y, AREA: batch_area})
        print("Epoch " + str(epoch) + ": Minibatch Loss= " + "{:.8f}".format(loss)  + ", %i Data Points/sec" % int(len(batch_x) / te))

        # log to list for plotting
        mse_mini_batch_list.append((epoch, loss))
        tmp = np.vstack(mse_mini_batch_list)
        plt.figure(figsize=(8,8))
        plt.semilogy(tmp[:,0], tmp[:,1])
        plt.xlabel('epoch')
        plt.ylabel('minibatch train loss')
        plt.savefig(CASE_NAME + '/minibatchloss.png')
        plt.close()

        T_CURRENT = time.time()
        logging.info("Epoch " + str(epoch) + ": Minibatch Loss= " + "{:.8f}".format(loss) + ", %i Data Points/sec" % int(len(batch_x) / te))
        logging.info('Time elapsed: ' + str((T_CURRENT - T_START) / 3600.0) + " Hours..")
        logging.info('\n')

    if epoch % plot_epoch == 0:
        # only draw the first snapshot
        up = sess.run(output_u, {X: TRAIN_DATA[:i_max, [1]], Y:TRAIN_DATA[:i_max, [2]], T: TRAIN_DATA[:i_max, [0]]})
        up = np.matmul(up, np.diag(std_tr[-DIM_U-1:-1])) + mean_tr[-DIM_U-1:-1]
        ut = TRAIN_DATA[:i_max, -DIM_U-1:-1]
        ut = np.matmul(ut, np.diag(std_tr[-DIM_U-1:-1])) + mean_tr[-DIM_U-1:-1]

        err = up - ut
        logging.info(str(err.shape))

        # draw the figure: prediction, truth, error
        fig, axs = plt.subplots(2, 3, figsize=(20, 8))
        im00 = axs[0,0].tricontourf(TRAIN_DATA[:i_max, 1] * std_tr[1] + mean_tr[1],
                                 TRAIN_DATA[:i_max, 2] * std_tr[2] + mean_tr[2],
                                 up[:,0], cmap='jet',
                                 levels=np.linspace(-2.5, 6, 50))
        # plt.colorbar(im0, ax=axs[1])
        im01 = axs[0,1].tricontourf(TRAIN_DATA[:i_max, 1] * std_tr[1] + mean_tr[1],
                                 TRAIN_DATA[:i_max, 2] * std_tr[2] + mean_tr[2],
                                 ut[:,0], cmap='jet',
                                 levels=np.linspace(-2.5, 6, 50))
        # plt.colorbar(im1, ax=axs[0])
        im02 = axs[0,2].tricontourf(TRAIN_DATA[:i_max, 1] * std_tr[1] + mean_tr[1],
                                 TRAIN_DATA[:i_max, 2] * std_tr[2] + mean_tr[2],
                                 err[:,0], cmap='jet',
                                 levels=np.linspace(-2.5, 6, 50))
        # plt.colorbar(im2, ax=axs[2])
        im10 = axs[1, 0].tricontourf(TRAIN_DATA[:i_max, 1] * std_tr[1] + mean_tr[1],
                                     TRAIN_DATA[:i_max, 2] * std_tr[2] + mean_tr[2],
                                     up[:, 1], cmap='jet',
                                     levels=np.linspace(-5, 5, 50))
        # plt.colorbar(im0, ax=axs[1])
        im11 = axs[1, 1].tricontourf(TRAIN_DATA[:i_max, 1] * std_tr[1] + mean_tr[1],
                                     TRAIN_DATA[:i_max, 2] * std_tr[2] + mean_tr[2],
                                     ut[:, 1], cmap='jet',
                                     levels=np.linspace(-5, 5, 50))
        # plt.colorbar(im11, ax=axs[0,1])
        im12 = axs[1, 2].tricontourf(TRAIN_DATA[:i_max, 1] * std_tr[1] + mean_tr[1],
                                     TRAIN_DATA[:i_max, 2] * std_tr[2] + mean_tr[2],
                                     err[:, 1], cmap='jet',
                                     levels=np.linspace(-5, 5, 50))
        # plt.colorbar(im2, ax=axs[2])
        axs[0,1].set_title('Truth')
        axs[0,0].set_title('Pred')
        axs[0,2].set_title('error')

        plt.savefig(CASE_NAME + '/pngs/true_pred_error_' + str(epoch) + '.png')
        plt.close()

    if epoch % checkpt_epoch == 0:
        # save model at check point
        tf.saved_model.simple_save(sess,
                                   CASE_NAME + '/saved_model_ckpt_' + str(epoch) + '/',
                                   inputs={"t": T, "x": X, "u": U},
                                   outputs={"eval_output_u": output_u})

    epoch += 1
logging.info("Optimization Finished!")
logging.info('Time elapsed: ', (T_END - T_START) / 3600.0, " Hours")
T_END = time.time()
