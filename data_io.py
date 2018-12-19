import numpy as np
import tensorflow as tf
import os
from config import cfg4_20 as cfg
from scipy.special import expit
import config

def hierarchical_compositional(genf_shape,n_samples,W=None,noise=None):
    "generate a bunch of samples, the activation function is frozen as relu"
    X = np.matrix(np.asarray(np.random.uniform(size=[n_samples,genf_shape[0]])))
    lastX = X
    # initialize
    if W is None:
        W = []
        for i in np.arange(len(genf_shape) - 1) + 1:
            w = np.matrix(np.random.uniform(low=-2, high=2, size=[genf_shape[i - 1], genf_shape[i]]))
            W.append(w)
    # apply weights, get Ys
    for i in np.arange(len(genf_shape) - 1):
        print "layer", i, "incoming shape", lastX.shape
        # print "test shape", lastX.shape,W[i].shape, (lastX * W[i]).shape, np.maximum(lastX * W[i],0).shape
        lastX = np.maximum(np.asarray(lastX * W[i]),0)
    Y = lastX
    if noise is not None:
        Y = Y + np.random.uniform(size=Y.shape,low=-0.5*noise,high=0.5*noise)
    return X,Y,W


def generate_dataset(out_path, f_shape, train_samples, test_samples, noise):
    if not os.path.exists(out_path): os.makedirs(out_path)
    # generate training set
    X_train,Y_train,W = hierarchical_compositional(f_shape, n_samples=train_samples, noise=noise)
    np.save(os.path.join(out_path, "X_train"), np.asarray(X_train,np.float32))
    np.save(os.path.join(out_path, "Y_train"), np.asarray(Y_train,np.float32))
    # generate testing set
    X_test,Y_test,_ = hierarchical_compositional(f_shape,n_samples=test_samples, W=W, noise=noise)
    np.save(os.path.join(out_path, "X_test"), np.asarray(X_test,np.float32))
    np.save(os.path.join(out_path, "Y_test"), np.asarray(Y_test,np.float32))
    print "data has been saved"


def read_dataset(db_path,batch_size):
    # load dataset
    X_train = np.load(os.path.join(db_path, "X_train.npy"))
    Y_train = np.load(os.path.join(db_path, "Y_train.npy"))
    X_test = np.load(os.path.join(db_path, "X_test.npy"))
    Y_test = np.load(os.path.join(db_path, "Y_test.npy"))
    print "loaded dataset of shapes:", X_train.shape,Y_train.shape,X_test.shape,Y_test.shape
    # make batches
    x_train, y_train = tf.train.slice_input_producer([X_train,Y_train])
    x_train, y_train = tf.train.batch([x_train,y_train],batch_size)
    x_test, y_test = tf.train.slice_input_producer([X_test, Y_test])
    x_test, y_test = tf.train.batch([x_test, y_test], batch_size)
    return x_train,y_train,x_test,y_test



if __name__ == "__main__":
    generate_dataset(cfg.db_path,cfg.genf_shape,cfg.train_samples,cfg.test_samples,noise=cfg.noise)
    x_train, y_train, x_test, y_test = read_dataset(cfg.db_path,cfg.batch_size)
    sess = tf.Session()
    coord = tf.train.Coordinator()
    sess.run(tf.global_variables_initializer())
    tf.train.start_queue_runners(sess,coord)
    _x_train, _y_train, _x_test, _y_test = sess.run([x_train, y_train, x_test, y_test])
    print "batch shapes:", _x_train.shape,_y_train.shape,_x_test.shape,_y_test.shape