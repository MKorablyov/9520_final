import time,os,sys
import tensorflow as tf
import numpy as np
from data_io import read_dataset,generate_dataset
import config
from matplotlib import pyplot as plt

def rescale(x, a, b, c, d):
    """
    Rescales variable from [a, b] to [c, d]
    """
    return c + ((d - c) / (b - a)) * (x - a)


def schwefel(x, xmin=-1, xmax=1):
    """
    https://www.sfu.ca/~ssurjano/schwef.html
    """
    x = rescale(x, xmin, xmax, -500, 500)
    #result = x.abs().sqrt().sin().mul(x).sum(1).mul(-1).add(418.9829 * x.size(1))
    result = 418.9829 * tf.to_float(tf.shape(x)[1]) - tf.reduce_sum(tf.sin(tf.abs(x)**0.5) * x,axis=1)
    return result

def perceptron_embedding(sizes,sess,coord):
    # init network (first 2 layers)
    # fixme changed the original implementation
    Ws = []
    top_layer = tf.get_variable("perc_embed",
                                shape=[sizes[0],sizes[1]],
                                initializer=tf.contrib.layers.xavier_initializer())
    Ws.append(top_layer)
    for i in range(1, len(sizes) - 1):
        name = "perceptron_fc" + str(i)
        shape = [sizes[i],sizes[i+1]]
        w = tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())
        top_layer = tf.nn.relu(tf.matmul(top_layer,w))
        Ws.append(w)
    return top_layer,Ws


def actcentron_embedding(sizes,sess,coord,output_range=3):
    # init network (first 2 layers)
    # fixme changed the original implementation
    Ws = []
    top_layer = tf.get_variable("perc_embed",
                                shape=[sizes[0],sizes[1]],
                                initializer=tf.contrib.layers.xavier_initializer())
    Ws.append(top_layer)
    for i in range(1, len(sizes) - 1):
        name = "perceptron_fc" + str(i)
        shape = [sizes[i],sizes[i+1]]
        w = tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())
        top_layer = tf.nn.relu(tf.matmul(top_layer,w))
        Ws.append(w)
    sess.run(tf.global_variables_initializer()) # FIXME: I need to initialize each of the weights separately
    tf.train.start_queue_runners(sess,coord)
    scaling = 1.7159 / tf.maximum(tf.abs(tf.reduce_max(top_layer)), tf.abs(tf.reduce_min(top_layer)))
    scaling_const = sess.run(scaling)
    act = output_range * tf.nn.tanh(scaling_const * top_layer)
    return act,Ws


def indep_embedding(sizes,sess,coord,output_range=3):
    # init network (first 2 layers)
    # fixme changed the original implementation
    Ws = []
    top_layer = tf.get_variable("perc_embed",
                                shape=[sizes[0],sizes[1]],
                                initializer=tf.contrib.layers.xavier_initializer())
    Ws.append(top_layer)
    #for i in range(1, len(sizes) - 1):
    #    name = "perceptron_fc" + str(i)
    #    shape = [sizes[i],sizes[i+1]]
    #    w = tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())
    #    top_layer = tf.nn.relu(tf.matmul(top_layer,w))
    #    Ws.append(w)
    sess.run(tf.global_variables_initializer()) # FIXME: I need to initialize each of the weights separately
    tf.train.start_queue_runners(sess,coord)
    scaling = 1.7159 / tf.maximum(tf.abs(tf.reduce_max(top_layer)), tf.abs(tf.reduce_min(top_layer)))
    scaling_const = sess.run(scaling)
    act = output_range * tf.nn.tanh(scaling_const * top_layer)
    return act,Ws



def embed_fc_tanh(layer_depths,layer_acts,sess,coord,output_range=3):
    "generic embedding of fully-connected layers and tg activation"
    # todo; use get_variable so that it is possible to run on the testing set
    # FIXME: I need to initialize each of the weights separately
    Ws = []
    # initialize layer 0 is special
    top_layer = tf.get_variable("embedding_fc_0",
                                shape=[layer_depths[0],layer_depths[1]],
                                initializer=tf.contrib.layers.xavier_initializer())
    Ws.append(top_layer)
    # initialize all other layers
    for i in range(1, len(layer_depths) - 1):
        name = "embedding_fc_" + str(i)
        shape = [layer_depths[i],layer_depths[i+1]]
        w = tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())
        top_layer = eval(layer_acts[i])(tf.matmul(top_layer,w))
        Ws.append(w)
    # initialize to help tanh nonlinearity to work
    sess.run(tf.global_variables_initializer())
    tf.train.start_queue_runners(sess,coord)
    scaling_const = sess.run(1.7159 / tf.maximum(tf.abs(tf.reduce_max(top_layer)), tf.abs(tf.reduce_min(top_layer))))
    top_layer = output_range * tf.nn.tanh(scaling_const * top_layer)
    return top_layer,Ws


def net3_embed1(X,fun_shape,em,em_shape,sess,coord):
    "3-layer network with embedding in the first layer"
    # [16,16,16,3]
    # W [batch, surface_dots, w_in, w_out]
    l0 = tf.expand_dims(X, 1)
    W1,PWs = eval(em)(em_shape,sess=sess,coord=coord) # [120, 60, 256]

    W1 = tf.reshape(W1,[1,em_shape[0],fun_shape[0],fun_shape[1]]) # 16,16

    l1 = tf.reduce_sum(tf.expand_dims(l0,3) * W1,axis=2)
    l1_act = tf.nn.relu(l1)
    W2 = tf.get_variable("W2", shape=[fun_shape[1], fun_shape[2]],
                         initializer=tf.contrib.layers.xavier_initializer(),trainable=False)
    l2 = tf.reduce_sum(tf.expand_dims(l1_act, 3) * W2, axis=2)
    l2_act = tf.nn.relu(l2)
    W3 = tf.get_variable("W3", shape=[fun_shape[2], fun_shape[3]],
                         initializer=tf.contrib.layers.xavier_initializer(),trainable=False)
    l3 = tf.reduce_sum(tf.expand_dims(l2_act, 3) * W3, axis=2)
    return X, l3, PWs + [W1,W2,W3]







def _try_params(n_iterations,batch_size,fun_shape,em,em_shape,db_path,lr,optimizer,scheduler):
    "try some parameters, report testing accuracy with square loss"
    # read data
    x_train, y_train, x_test, y_test = read_dataset(db_path, batch_size)
    # initialize training/testing graph
    # initialize session
    sess = tf.Session()
    coord = tf.train.Coordinator()
    _,yhat_train,Ws = net3_embed1(X=x_train,fun_shape=fun_shape,em=em,em_shape=em_shape,sess=sess,coord=coord)
    y_diff = tf.expand_dims(y_train,1) - yhat_train

    train_loss = tf.reduce_mean(tf.reduce_mean(y_diff**2,axis=2),axis=0)
    lr_current = tf.placeholder(tf.float32)
    train_step = eval(optimizer)(learning_rate=lr_current).minimize(train_loss)

    sess.run(tf.global_variables_initializer())
    tf.train.start_queue_runners(sess,coord)
    _train_losses = []
    for i in range(n_iterations):
        _train_loss,_ = sess.run([train_loss,train_step],feed_dict={lr_current:lr})
        _train_losses.append(_train_loss)

        if scheduler=="none":
            pass
        elif scheduler == "dlp":
            # scheduler the step size
            if i % 2000 == 1999:
                if np.mean(_train_losses[-1000:]) >= np.mean(_train_losses[-2000:-1000]):
                    lr = lr * 0.5
        else:
            raise ValueError("unknown scheduler")

        # printing
        if i % 100 == 1:
            # print "argmin of the train loss", _y_diff[_train_loss.argmin()],
            print "step:",i,"mean_loss", np.mean(_train_loss), "min_loss", np.min(_train_loss),
            # print "y_train:", np.mean(_y_train), np.var(_y_train),_y_train.shape,
            # print "y_hat:", np.mean(_yhat_train),np.var(_yhat_train), _yhat_train.shape,
            print "lr",lr
        # history
        if i % 100 == 1:
            _train_loss, = sess.run([train_loss],feed_dict={lr_current:lr})
    sess.close()
    tf.reset_default_graph()
    sel_point = np.mean(_train_losses[-200:-100], axis=0).argmin()
    minmean_loss = np.mean(_train_losses[:-100], axis=0)[sel_point]
    loss_hist = np.asarray(_train_losses)[:,sel_point]
    return minmean_loss,loss_hist


def try_params(n_iterations,batch_size,fun_shape,em,em_shape,db_path,lr,optimizer,scheduler):
    train_cost, train_cost_hist = tf.py_func(_try_params,
                                            [n_iterations,batch_size,fun_shape,em,em_shape,db_path,lr,optimizer,scheduler],
                                            [tf.float32,tf.float32])
    sess = tf.Session()
    _train_cost,_train_cost_hist = sess.run([train_cost,train_cost_hist])
    sess.close()
    return _train_cost,_train_cost_hist


if __name__ == "__main__":
    # set up the config and folders
    config_name = "cfg4_32"
    if len(sys.argv) >= 2:
        config_name = sys.argv[1]
    cfg = eval("config." + config_name)
    if not os.path.exists(cfg.out_path): os.makedirs(cfg.out_path)
    for lr in cfg.lrs:
        train_costs = []
        train_cost_hists = []
        for i in range(cfg.n_runs):
            # generate dataset
            #generate_dataset(cfg.db_path, cfg.genf_shape, cfg.train_samples, cfg.test_samples, noise=cfg.noise)
            #train_cost,train_cost_hist = try_params(1000,cfg.batch_size,[64,32,1],cfg.db_path,cfg.test_samples, lr=lr)
            train_cost, train_cost_hist = try_params(n_iterations= cfg.n_iterations,
                                                     batch_size= cfg.batch_size,
                                                     fun_shape= cfg.fun_shape,
                                                     em=cfg.em,
                                                     em_shape=cfg.em_shape,
                                                     db_path=cfg.db_path, lr=lr, optimizer=cfg.optimizer,scheduler=cfg.scheduler)
            print "\n", "train cost", train_cost
            train_costs.append(train_cost)
            train_cost_hists.append(train_cost_hist)

        train_costs = np.asarray(train_costs)
        train_cost_hists = np.asarray(train_cost_hists)
        train_cost_hist = np.mean(train_cost_hists,axis=0)
        np.savetxt(os.path.join(cfg.out_path,config_name + "_train_cost_hist_lr_" + str(lr)),train_cost_hist)


