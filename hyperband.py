import numpy as np
from random import random
from math import log, ceil
from time import time, ctime
from hyperopt import hp #Install Hyperopt
from hyperopt.pyll.stochastic import sample


#TODO: Move End2End example here
#TODO: Use reporting from Kfirs

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


def try_params(n_iterations, params,data,model):
    max_iter = int(round(n_iterations * 1))
    print "max_iter:", max_iter
    x_train_ = data['x_train'].astype(np.float64)
    x_test_ = data['x_test'].astype(np.float64)
    y_train_ = data['y_train'].copy()
    y_test_ = data['y_test'].copy()
    local_data = {'x_train': x_train_, 'y_train': y_train_,
                  'x_test': x_test_, 'y_test': y_test_}
    params_ = dict(params)
    clf = model(max_iter=max_iter) #Adapt this to suit your own models
    return clf.evaluate(local_data) #this should train your model and return the loss in this form -  {'loss': '', 'auc': auc}



class Hyperband:

    def __init__(self, get_params_function, try_params_function):
        self.get_params = get_params_function
        self.try_params = try_params_function

        self.max_iter = 81  # maximum iterations per configuration
        self.eta = 3  # defines configuration downsampling rate (default = 3)

        self.logeta = lambda x: log(x) / log(self.eta)
        self.s_max = int(self.logeta(self.max_iter))
        self.B = (self.s_max + 1) * self.max_iter

        self.results = []  # list of dicts
        self.counter = 0
        self.best_loss = np.inf
        self.best_counter = -1

    
    def run(self, skip_last=0, dry_run=False):

        for s in reversed(range(self.s_max + 1)):

            # initial number of configurations
            n = int(ceil(self.B / self.max_iter / (s + 1) * self.eta ** s))

            # initial number of iterations per config
            r = self.max_iter * self.eta ** (-s)

            # n random configurations
            T = [self.get_params() for i in range(n)]

            for i in range((s + 1) - int(skip_last)):  # changed from s + 1

                # Run each of the n configs for <iterations>
                # and keep best (n_configs / eta) configurations

                n_configs = n * self.eta ** (-i)
                n_iterations = r * self.eta ** (i)

                print "\n*** {} configurations x {:.1f} iterations each".format(
                    n_configs, n_iterations)

                val_losses = []
                early_stops = []

                for t in T:

                    self.counter += 1
                    print "\n{} | {} | lowest loss so far: {:.4f} (run {})\n".format(
                        self.counter, ctime(), self.best_loss, self.best_counter)

                    start_time = time()

                    if dry_run:
                        result = {'loss': random(), 'log_loss': random(), 'auc': random()}
                    else:
                        result = self.try_params(n_iterations, t)  # <---

                    assert (type(result) == dict)
                    assert ('loss' in result)

                    seconds = int(round(time() - start_time))
                    print "\n{} seconds.".format(seconds)

                    loss = result['loss']
                    val_losses.append(loss)

                    early_stop = result.get('early_stop', False)
                    early_stops.append(early_stop)

                    # keeping track of the best result
                    if loss < self.best_loss:
                        self.best_loss = loss
                        self.best_counter = self.counter

                    result['counter'] = self.counter
                    result['seconds'] = seconds
                    result['params'] = t
                    result['iterations'] = n_iterations

                    self.results.append(result)

                # select a number of best configurations for the next loop
                # filter out early stops, if any
                indices = np.argsort(val_losses)
                T = [T[i] for i in indices if not early_stops[i]]
                T = T[0:int(n_configs / self.eta)]
        return self.results