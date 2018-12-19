import numpy as np
from random import random
from math import log, ceil
from time import time, ctime
from hyperopt import hp #Install Hyperopt
from hyperopt.pyll.stochastic import sample


space = {
    'recurrent_cell_size': hp.quniform( 'c', 80, 160, 20),
    'batch_size':hp.quniform('b',15,50,10)
}

def handle_integers(params):
    new_params = {}
    for k, v in params.items():
        if type(v) == float and int(v) == v:
            new_params[k] = int(v)
        else:
            new_params[k] = v
    return new_params

def get_params():
    params = sample(space)
    return handle_integers(params)




# def try_params(n_iterations, params, data, model):
#     max_iter = int(round(n_iterations * 1))
#     print "max_iter:", max_iter
#     x_train_ = data['x_train'].astype(np.float64)
#     x_test_ = data['x_test'].astype(np.float64)
#     y_train_ = data['y_train'].copy()
#     y_test_ = data['y_test'].copy()
#     local_data = {'x_train': x_train_, 'y_train': y_train_,'x_test': x_test_, 'y_test': y_test_}
#     #params_ = dict(params)
#     clf = model(max_iter=max_iter) #Adapt this to suit your own models
#     return clf.evaluate(local_data) #this should train your model and return the loss in this form -  {'loss': '', 'auc': auc}








for i in range(100):
    print get_params()