import time
import tensorflow as tf
import numpy as np
from data_io import read_dataset
from config import cfg


def hierarchical_compositional(X,f_shape,W=None):
    "wraps a funciton of shape f with sigmoid activation around X"
    # initialize weights when needed
    if W is None:
        W = []
        for i in np.arange(len(f_shape)-1)+1:
            shape = [f_shape[i-1],f_shape[i]]
            w = tf.get_variable("hc" + str(i), shape=shape, initializer=tf.contrib.layers.xavier_initializer())
            W.append(w)
    # apply the function
    lastX = X
    for w in W:
        lastX = tf.nn.relu(tf.matmul(lastX,w))
    Y = lastX
    return X,Y,W


def _try_params(n_iterations,batch_size,f_shape,db_path,test_samples,lr):
    "try some parameters, report testing accuracy with square loss"
    # read data
    x_train, y_train, x_test, y_test = read_dataset(db_path, batch_size)
    # initialize training/testing graph
    _,yhat_train,W = hierarchical_compositional(x_train,f_shape)
    train_loss = tf.reduce_mean((y_train - yhat_train)**2)

    #train_loss = - tf.reduce_mean(tf.sin(y_test - yhat_train) / (y_test - yhat_train))  # FIXME another loss

    shuffle_loss = tf.reduce_mean((tf.random_shuffle(y_train) - yhat_train)**2)
    train_step = tf.train.AdamOptimizer(learning_rate=lr).minimize(train_loss)
    _,yhat_test,_ = hierarchical_compositional(x_train, f_shape,W)
    test_loss = tf.reduce_mean((y_test - yhat_test) ** 2)

    # initialize session
    sess = tf.Session()
    coord = tf.train.Coordinator()
    sess.run(tf.global_variables_initializer())
    tf.train.start_queue_runners(sess,coord)
    # train the network
    for i in range(n_iterations):
        _train_loss, _shuffle_loss, _ = sess.run([train_loss, shuffle_loss, train_step])
        if i %100 ==1:
            print "step:", i,"%.3f"%_train_loss,"%.3f"%_shuffle_loss
    # test the network
    test_iterations = int(test_samples / batch_size)
    train_costs = []
    test_costs = []
    for i in range(test_iterations):
        _train_loss,_test_loss = sess.run([train_loss,test_loss])
        train_costs.append(_train_loss)
        test_costs.append(_test_loss)
    train_cost = np.mean(np.asarray(train_costs))
    test_cost = np.mean(np.asarray(test_costs))
    sess.close()
    tf.reset_default_graph()
    return train_cost,test_cost


def try_params(n_iterations,batch_size,f_shape,db_path,test_samples,lr):
    train_cost,test_cost = tf.py_func(_try_params,
                                      [n_iterations,batch_size,f_shape,db_path,test_samples,lr],
                                      [tf.float32,tf.float32])
    sess = tf.Session()
    train_cost,test_cost = sess.run([train_cost,test_cost])
    sess.close()
    return train_cost,test_cost


if __name__ == "__main__":
    train_cost, test_cost = try_params(20000,cfg.batch_size,[64,32,1],cfg.db_path,cfg.test_samples, lr=0.0001)
    print "\n", "-7,5- pars:", train_cost, test_cost
    # train_cost,test_cost = _try_params(20000,cfg.batch_size,[10,20,1],cfg.db_path,cfg.test_samples)
    # print "\n","-70,50- pars:", train_cost, test_cost
    # train_cost,test_cost = _try_params(20000,cfg.batch_size,[10,200,1],cfg.db_path,cfg.test_samples)
    # print "\n","-700,500- pars:", train_cost, test_cost
    # train_cost,test_cost = _try_params(20000,cfg.batch_size,[10,2000,1],cfg.db_path,cfg.test_samples)
    # print "\n","-7000,5000- pars:", train_cost, test_cost