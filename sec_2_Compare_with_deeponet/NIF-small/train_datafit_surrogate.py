
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
# import GPUtil

import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import time
import logging
# from batchup import data_source
import os
import argparse

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# DEVICE_ID_LIST = GPUtil.getFirstAvailable(maxLoad=0.05, maxMemory=0.05)
# os.environ["CUDA_VISIBLE_DEVICES"] = str(DEVICE_ID_LIST[0])

parser = argparse.ArgumentParser(description='config')
parser.add_argument('--TRAIN_DATA',type=str, help='normalized train data file path')
parser.add_argument('--NETWORK_TYPE',type=str, help='model type')
parser.add_argument('--N_S',type=int, help='width of shapenet/MLP')
parser.add_argument('--RANK',type=int, help='number of ranks')
parser.add_argument('--N_T',type=int, help='width of parameternet')
parser.add_argument('--ACT',type=str, help='activation function type')
parser.add_argument('--L_R',type=float, help='learning rate')
parser.add_argument('--BATCH_SIZE',type=int, help='batch size')
parser.add_argument('--EPOCH',type=int, help='total epoch')
args = parser.parse_args()

TRAIN_DATA = np.load(args.TRAIN_DATA)['data']
NETWORK_TYPE = args.NETWORK_TYPE
N_t = args.N_T
N_s = args.N_S
N_p = args.RANK
ACT_STR = args.ACT
learning_rate = args.L_R
batch_size = args.BATCH_SIZE
nepoch = args.EPOCH 

if ACT_STR == 'swish':
    ACT = tf.nn.swish
elif ACT_STR == 'tanh':
    ACT = tf.nn.tanh
elif ACT_STR == 'relu':
    ACT = tf.nn.relu


gpu_options = tf.GPUOptions(allow_growth=True) 

logging.basicConfig(filename='log',level=logging.INFO,format='%(message)s')

# print(tf.__version__)

# Import MNIST data
# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Training Parameters
num_gpus = 1
# nepoch = 40001
# learning_rate = 1e-3
# batch_size = 32768 # 2048 # 512
display_epoch = 100
checkpt_epoch = nepoch - 1

# Shape Network Parameters
# N_s = 100 # 100
L_s = 3

## TMP
if NETWORK_TYPE == 'DEEPONET':
    Total_para = (L_s-2)*(N_s+1)*N_s + 2*N_s + 1 + N_p + (N_s + 1)*N_p
else:
    Total_para = (L_s-1)*(N_s+1)*N_s + 3*N_s + 1

logging.info('total number of shape net = ' + str(Total_para))

# Logging hyperparameters
logging.info("Number of GPUs = " + str(num_gpus))
logging.info("Total epoch = " + str(nepoch))
logging.info("Learnign rate = " + str(learning_rate))
logging.info("Batch size = " + str(batch_size))
logging.info("\n")


# Build a convolutional neural network
init_weight = tf.truncated_normal_initializer(stddev=1e-1)

if NETWORK_TYPE == 'MLP':
    def NETWORK(t, x, reuse):
        # Define a scope for reusing the variables
        with tf.variable_scope('NIN', reuse=reuse):
               
            u = tf.concat([t, x], axis=-1)
            u = tf.layers.dense(u, N_s, activation=ACT,
                                kernel_initializer=init_weight, bias_initializer=init_weight)
            u = tf.layers.dense(u, N_s, activation=ACT,
                                kernel_initializer=init_weight, bias_initializer=init_weight) + u
            u = tf.layers.dense(u, N_s, activation=ACT,
                                kernel_initializer=init_weight, bias_initializer=init_weight) + u
            u = tf.layers.dense(u, 1, activation=None,
                                kernel_initializer=init_weight, bias_initializer=init_weight)

        return u

elif NETWORK_TYPE == 'DEEPONET':
    def NETWORK(t, x, reuse):
        # Define a scope for reusing the variables
        with tf.variable_scope('NIN', reuse=reuse):

            # Build "parameter net"
            para_net = t # tf.concat([t],axis=-1)
            para_net = tf.layers.dense(para_net, N_t, activation=ACT,
                                       kernel_initializer=init_weight, bias_initializer=init_weight)
            para_net = tf.layers.dense(para_net, N_t, activation=ACT,
                                       kernel_initializer=init_weight, bias_initializer=init_weight)
            # para_net = tf.layers.dense(para_net, 1,  activation=ACT,
            #                            kernel_initializer=init_weight, bias_initializer=init_weight)
            para_net = tf.layers.dense(para_net, N_p+1, activation=None,
                                       kernel_initializer=init_weight, bias_initializer=init_weight)
            # # Collect para-net
            # para_net = tf.identity(para_net, name="para_net")

            phi_x = tf.layers.dense(x, N_s, activation=ACT, kernel_initializer=init_weight, bias_initializer=init_weight)
            phi_x = phi_x + tf.layers.dense(phi_x, N_s, activation=ACT, kernel_initializer=init_weight, bias_initializer=init_weight)
            phi_x = tf.layers.dense(phi_x, N_p, activation=ACT, kernel_initializer=init_weight, bias_initializer=init_weight)

            # final layer
            weight_final = para_net[:,0:N_p]
            weight_final = tf.reshape(weight_final, shape=[-1, N_p,  1])
            bias_final = para_net[:,-1]
            bias_final = tf.reshape(bias_final, shape=[-1, 1])

            u = tf.einsum('ai,aij->aj', phi_x, weight_final) + bias_final

        return u

elif NETWORK_TYPE == 'NIF':
    def NETWORK(t, x, reuse):
        # Define a scope for reusing the variables
        with tf.variable_scope('NIN', reuse=reuse):

            # Build "parameter net"
            para_net = t # tf.concat([t],axis=-1)
            para_net = tf.layers.dense(para_net, N_t, activation=ACT,
                                       kernel_initializer=init_weight, bias_initializer=init_weight)
            para_net = tf.layers.dense(para_net, N_t, activation=ACT,
                                       kernel_initializer=init_weight, bias_initializer=init_weight)
            para_net = tf.layers.dense(para_net, 1,  activation=ACT,
                                       kernel_initializer=init_weight, bias_initializer=init_weight)
            para_net = tf.layers.dense(para_net, Total_para, activation=None,
                                       kernel_initializer=init_weight, bias_initializer=init_weight)
            # # Collect para-net
            # para_net = tf.identity(para_net, name="para_net")

            # Distribute to weight and biases
            weight_1 = para_net[:,0:N_s];                           weight_1 = tf.reshape(weight_1, shape=[-1, 1,  N_s])
            weight_2 = para_net[:,N_s:((N_s+1)*N_s)];               weight_2 = tf.reshape(weight_2, shape=[-1, N_s, N_s])
            weight_3 = para_net[:,(N_s**2+N_s):(2*N_s**2+N_s)];     weight_3 = tf.reshape(weight_3, shape=[-1, N_s, N_s])
            weight_4 = para_net[:,(2*N_s**2+N_s):(2*N_s**2+2*N_s)]; weight_4 = tf.reshape(weight_4, shape=[-1, N_s, 1 ])
            bias_1   = para_net[:,(2*N_s**2+2*N_s):(2*N_s**2+3*N_s)]; bias_1   = tf.reshape(bias_1,   shape=[-1, N_s])
            bias_2   = para_net[:,(2*N_s**2+3*N_s):(2*N_s**2+4*N_s)]; bias_2   = tf.reshape(bias_2,   shape=[-1, N_s])
            bias_3   = para_net[:,(2*N_s**2+4*N_s):(2*N_s**2+5*N_s)]; bias_3   = tf.reshape(bias_3,   shape=[-1, N_s])
            bias_4   = para_net[:,(2*N_s**2+5*N_s):];                 bias_4   = tf.reshape(bias_4,   shape=[-1, 1])

            # Build "shape net"
            u = ACT(tf.einsum('ai,aij->aj', x, weight_1) + bias_1)
            u = ACT(tf.einsum('ai,aij->aj', u, weight_2) + bias_2) + u
            u = ACT(tf.einsum('ai,aij->aj', u, weight_3) + bias_3) + u
            u = tf.einsum('ai,aij->aj',u, weight_4) + bias_4

        return u

def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


# By default, all variables will be placed on '/gpu:0'
# So we need a custom device function, to assign all variables to '/cpu:0'
# Note: If GPUs are peered, '/gpu:0' can be a faster option
PS_OPS = ['Variable', 'VariableV2', 'AutoReloadVariable']

def assign_to_device(device, ps_device='/cpu:0'):
    def _assign(op):
        node_def = op if isinstance(op, tf.NodeDef) else op.node_def
        if node_def.op in PS_OPS:
            return "/" + ps_device
        else:
            return device

    return _assign
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

# Place all ops on CPU by default
with tf.device('/cpu:0'):
    tower_grads = []
    reuse_vars = False

    # tf Graph input
    # X = tf.placeholder(tf.float32, [None, num_input])
    # Y = tf.placeholder(tf.float32, [None, num_classes])
    X  = tf.placeholder(tf.float32, [None, 1],name='input_X')
    # NU = tf.placeholder(tf.float32, [None, 1],name='input_NU')
    T  = tf.placeholder(tf.float32, [None, 1],name='input_T')
    U  = tf.placeholder(tf.float32, [None, 1])


    
    # Loop over all GPUs and construct their own computation graph
    for i in range(num_gpus):
        # with tf.device(assign_to_device('/cpu:{}'.format(i), ps_device='/cpu:0')):
        with tf.device(assign_to_device('/gpu:{}'.format(i), ps_device='/cpu:0')):

            # Split data between GPUs
            _x  = X[ i * batch_size: (i+1) * batch_size]
            # _nu = NU[i * batch_size: (i+1) * batch_size]
            _t  = T[ i * batch_size: (i+1) * batch_size]
            _u  = U[ i * batch_size: (i+1) * batch_size]

            # Because Dropout have different behavior at training and prediction time, we
            # need to create 2 distinct computation graphs that share the same weights.

            # Create a graph for training
            output_u = NETWORK(_t, _x, reuse=reuse_vars)
    
            # Define loss and optimizer (with train logits, for dropout to take effect)
            loss_op = tf.reduce_mean(tf.square(output_u - _u)) # + l2_loss
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            grads = optimizer.compute_gradients(loss_op)

            reuse_vars = True # it means only first GPU creates the net, the rest share with it
            tower_grads.append(grads)

    # collect all grads from GPUs, average them
    tower_grads = average_gradients(tower_grads)
    train_op = optimizer.apply_gradients(tower_grads)

    # Only first GPU compute accuracy
    # Create the same graph for evaluation
    eval_output_u = NETWORK(T, X, reuse=True)
    eval_output_u = tf.identity(eval_output_u, name='output_u')
    # para_net= tf.identity(para_net, name='para_net')

    # Evaluate model (with test logits, for dropout to be disabled)
    loss_op_all = tf.reduce_mean(tf.square(eval_output_u - U)) # + l2_loss

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()
    
    # init = tf.initializers.variance_scaling()
    # prepare data
    # TRAIN_DATA = np.load('ks_train_n_0.npz')['data']
    # ds = data_source.ArrayDataSource([TRAIN_DATA[:,[0]], TRAIN_DATA[:,[1]], TRAIN_DATA[:,[2]], TRAIN_DATA[:,[3]]])
    
    # Error
    TRAIN_MSE_LIST = []

    # Start Training
    # with tf.Session() as sess:
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,gpu_options=gpu_options))

    # Run the initializer
    sess.run(init)
    T_START = time.time()

    ## print the total number of trainable parameters
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        # print(shape)
        # print(len(shape))
        variable_parameters = 1
        for dim in shape:
            # print(dim)
            variable_parameters *= dim.value
        # print(variable_parameters)
        total_parameters += variable_parameters
    logging.info('total trainable = ' + str(total_parameters))

    # Keep training until reach max iterations
    for epoch in range(1, nepoch + 1):
        total_batch_size = num_gpus * batch_size
        # for (batch_nu, batch_x, batch_t, batch_u) in ds.batch_iterator(batch_size=total_batch_size,
        #                                                                shuffle=np.random.RandomState(epoch)):
        for batch in iterate_minibatches(TRAIN_DATA[:,0:2], TRAIN_DATA[:,-1], batch_size, shuffle=True):
            feature_batch, batch_u = batch
            # batch_nu = feature_batch[:,[0]]
            batch_x  = feature_batch[:,[0]]
            batch_t  = feature_batch[:,[1]]
            batch_u = batch_u.reshape(-1,1)            

            ts = time.time()
            # Get a batch for each GPU
            # batch_x, batch_y = mnist.train.next_batch(batch_size * num_gpus)

            # Run optimization op (backprop)
            sess.run(train_op, feed_dict={X: batch_x, T:batch_t, U: batch_u})
            te = time.time() - ts

        if epoch % display_epoch == 0 or epoch == 1:
            # Calculate batch loss and accuracy
            loss = sess.run([loss_op], feed_dict={X: batch_x, T:batch_t, U: batch_u})[0]
            loss_all = sess.run([loss_op_all], feed_dict={X: TRAIN_DATA[:,[0]], T:TRAIN_DATA[:,[1]], U: TRAIN_DATA[:,[2]]})[0]
            # print("Epoch " + str(epoch) + ": Minibatch Loss= " + "{:.8f}".format(loss) +" Total Loss ="+ "{:.8f}".format(loss_all) +  ", %i Data Points/sec" % int(len(batch_x)/te))
            # Record Train mse loss
            TRAIN_MSE_LIST.append(loss_all)

            # Log to file
            T_CURRENT = time.time()
            logging.info("Epoch " + str(epoch) + ": Minibatch Loss= " + "{:.8f}".format(loss) +" Total Loss ="+ "{:.8f}".format(loss_all) +  ", %i Data Points/sec" % int(len(batch_x)/te))
            logging.info('Time elapsed: '+str((T_CURRENT-T_START)/3600.0)+" Hours..")
            # logging.info('\n')

            plt.figure()
            plt.semilogy(np.array(TRAIN_MSE_LIST))
            plt.xlabel('epoch')
            plt.ylabel('MSE loss')
            plt.savefig('loss.png')
            plt.close()

        if epoch % checkpt_epoch == 0:
            # save model at check point
            tf.saved_model.simple_save(sess,
                                       './saved_model_ckpt_'+str(epoch)+'/',
                                       inputs={"t": T, "x": X, "u": U},
                                       outputs={"eval_output_u": eval_output_u})

        epoch += 1
    logging.info("Optimization Finished!")
    T_END = time.time()

    logging.info('Time elapsed: ' + str((T_END-T_START)/3600.0)+ " Hours")
    # End of training: save loss and save reconstruction

    logging.info('Training data size = '+ str(TRAIN_DATA.shape) )
    u_rec = sess.run([eval_output_u], feed_dict={X: TRAIN_DATA[:,[0]], T:TRAIN_DATA[:,[1]], U: TRAIN_DATA[:,[2]]})[0]
    # print(u_rec)
    # print(u_rec.shape)
    np.savez('mse_saved.npz',loss=np.array(TRAIN_MSE_LIST), time=T_END-T_START, u_rec=u_rec)

