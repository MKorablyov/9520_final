import matplotlib
matplotlib.use('Agg')
import time
import tensorflow as tf
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
import os,sys
import config
from data_io import generate_dataset
from some_model import try_params

# generate the dataset
# [64,1,1] [64, 64, 1] (1,2,4,8,16,32,64)
# [64,1,64] [64, 64, 64]
# [7x7 table]

# set up the config and folders
config_name = "cfg21"
if len(sys.argv) >= 2:
    config_name = sys.argv[1]
cfg = eval("config." + config_name)
if not os.path.exists(cfg.out_path): os.makedirs(cfg.out_path)


def param_test(cfg):
    train_cost_means = []
    train_cost_vars = []
    test_cost_means = []
    test_cost_vars = []

    for noise in cfg.noises:
        for i in range(len(cfg.batch_sizes)):
            batch_size = cfg.batch_sizes[i]
            lr = cfg.lrs[i]
            train_steps = cfg.train_steps[i]
            train_costs = []
            test_costs = []
            for i in range(cfg.n_repeats):
                generate_dataset(cfg.db_path, [64, cfg.l2_val, cfg.l3_val], cfg.train_samples, cfg.test_samples,noise)
                train_cost, test_cost = try_params(train_steps, batch_size, [64, cfg.l2_val, cfg.l3_val], cfg.db_path,
                                                   cfg.test_samples,lr=lr)
                train_costs.append(train_cost)
                test_costs.append(test_cost)
            train_cost_mean = np.mean(train_costs)
            train_cost_var = np.var(train_costs)
            test_cost_mean = np.mean(test_costs)
            test_cost_var = np.var(test_costs)
            train_cost_means.append(train_cost_mean)
            train_cost_vars.append(train_cost_var)
            test_cost_means.append(test_cost_mean)
            test_cost_vars.append(test_cost_var)
    # train_cost_means = np.asarray(np.arange(49),np.float32)
    # train_cost_vars = np.asarray(np.arange(49),np.float32)
    # test_cost_means = np.asarray(np.arange(49),np.float32)
    # test_cost_vars = np.asarray(np.arange(49),np.float32)
    # plots
    runs_shape = [cfg.l2_vals.shape[0], cfg.l3_vals.shape[0]]
    fig, axes = plt.subplots(2, 3, sharex=True, sharey=True)  # fig, axes = plt.subplots(2, 2,sharex=True,sharey=True)
    fig.set_size_inches(19.2, 12.8)
    ax = fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    plt.xlabel("layer 2 depth", fontsize=18)
    plt.ylabel("layer 3 depth", fontsize=18)
    # plot mean accuracy
    train_cost_means = np.around(np.reshape(np.asarray(train_cost_means), runs_shape), decimals=3)
    df = pd.DataFrame(data=train_cost_means, index=cfg.l2_vals, columns=cfg.l3_vals)
    sns.heatmap(df, cmap="YlGnBu", annot=True, cbar=False, ax=axes[0, 0])
    axes[0, 0].set_title("mean train loss")
    test_cost_means = np.around(np.reshape(np.asarray(test_cost_means), runs_shape), decimals=3)
    df = pd.DataFrame(data=test_cost_means, index=cfg.l2_vals, columns=cfg.l3_vals)
    sns.heatmap(df, cmap="YlGnBu", annot=True, cbar=False, ax=axes[0, 1])
    axes[0, 1].set_title("mean test loss")
    diff_cost = test_cost_means - train_cost_means
    df = pd.DataFrame(data=diff_cost, index=cfg.l2_vals, columns=cfg.l3_vals)
    sns.heatmap(df, cmap="YlGnBu", annot=True, cbar=False, ax=axes[0, 2])
    axes[0, 2].set_title("mean (test-train) loss")
    # plot var accuracy
    train_cost_vars = np.around(np.reshape(np.asarray(train_cost_vars), runs_shape), decimals=3)
    df = pd.DataFrame(data=train_cost_vars, index=cfg.l2_vals, columns=cfg.l3_vals)
    sns.heatmap(df, cmap="YlGnBu", annot=True, cbar=False, ax=axes[1, 0])
    axes[1, 0].set_title("std train loss")
    test_cost_vars = np.around(np.reshape(np.asarray(test_cost_vars), runs_shape), decimals=3)
    df = pd.DataFrame(data=test_cost_vars, index=cfg.l2_vals, columns=cfg.l3_vals)
    sns.heatmap(df, cmap="YlGnBu", annot=True, cbar=False, ax=axes[1, 1])
    axes[1, 1].set_title("std test loss")
    diff_cost = test_cost_vars - train_cost_vars
    df = pd.DataFrame(data=diff_cost, index=cfg.l2_vals, columns=cfg.l3_vals)
    sns.heatmap(df, cmap="YlGnBu", annot=True, cbar=False, ax=axes[1, 2])
    axes[1, 2].set_title("std (train-test) loss")
    plt.savefig(os.path.join(cfg.out_path, "combined"))
    plt.close()

if __name__ == "__main__":
    param_test(cfg)


# 1. I want to show that I can recover a polynomial today! 6th
# 2. In my old idea: it would be interesting to try to learn some dumper over Adam (shave sharp grads as an example)
# 3. Would be interesting to compare wavelets and convolutions

# some ideas: I might be able to prove the MEMO if I try
# the limited number of functions F on the training data (MNIST)
# and show that without training some of the functions may be quite good

# it is better to optimize over the space of structures of polynomials (which is an even larger space)
# I thought about that in the morning that GD search is different from random search
# can I prove that random search produces qualitatively different results?



# let's save the dataset to the disk; I will have to do some stuff with real data anyway

# the coolest idea so far:
# I can generate a controlled distribution of wavelets


# findings:
# 1. we can't fit the polynomial with exact shape when we don't have enough datapoints
# proof for 1-layer net (or, maybe, we can?)
# f_shape = [10, 50, 1]
# train_samples = 100
# test_samples = 10000


# 2. we can't fit polynomial with the exact shape because our optimization is non-convex
# I am not sure if I can prove that ??


# 3. Wavelet fold fits convolution, and convolution fits wavelet fold
# how each thing fits the other thing
# can I fit a wavelet to itself ??
# 4. Protein folding with wavelet
# or prove that the wavelet is better for representing the wavelet


