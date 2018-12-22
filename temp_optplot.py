import os,sys,time
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter
# plot to compare optimizers
# 30 tf.train.GradientDescentOptimizer (with scheduler)
# 32 tf.train.AdamOptimizer
# 34 tf.train.RMSPropOptimizer
# 36 tf.train.AdadeltaOptimizer
# 38 tf.train.GradientDescentOptimizer


db_root = "/home/maksym/Desktop/slt_titan"
out_path = "/home/maksym/Desktop/9520_final/plots"
#folders = ["cfg4_30","cfg4_31","cfg4_32","cfg4_33","cfg4_34","cfg4_35","cfg4_36","cfg4_37","cfg4_38","cfg4_39"]
folders= ["cfg4_29","cfg4_30","cfg4_31"]

#from os import listdir
#from os.path import isfile, join


runs_means = []
runs_filts = []
for folder_name in folders:
    folder = os.path.join(db_root, folder_name)
    runfiles = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    runs_data = []
    for runfile in runfiles:
        run_data = np.loadtxt(os.path.join(folder,runfile))
        runs_data.append(run_data)
    # work on data
    runs_data = np.asarray(runs_data)
    runs_mean = np.mean(runs_data,axis=0)
    runs_filt = savgol_filter(runs_mean,101,4)
    runs_means.append(runs_mean)
    runs_filts.append(runs_filt)



X = np.arange(0,25000) * 0.001

fig = plt.figure()
matplotlib.rcParams.update({'font.size': 18})
fig.set_size_inches(12.8, 12.8)
ax = fig.gca(yscale="log")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")



# plot SGD
plt.plot(X,runs_filts[0]) # color='c', alpha=0.3
plt.plot(X,runs_filts[1][:25000])
# plot ADAM
plt.plot(X,runs_filts[2][:25000])


ax.legend(["SGD-1x256","SGD-64x256","SGD-256x256"])#,prop={'size': 12})
plt.savefig(os.path.join(out_path,  "optimizers2.png"))
plt.close()



# # plot SGD
# plt.plot(X,runs_filts[8],color="C0") # color='c', alpha=0.3
# plt.plot(X,runs_filts[9],color="C0",alpha=0.5)
# # plot ADAM
# plt.plot(X,runs_filts[2],color="C1")
# plt.plot(X,runs_filts[3],color="C1",alpha=0.5)
# # plot RMSprop
# plt.plot(X,runs_filts[4],color="C2")
# plt.plot(X,runs_filts[5],color="C2",alpha=0.5)
# # plot AdaDelta
# plt.plot(X,runs_filts[6],color="C3")
# plt.plot(X,runs_filts[7],color="C3",alpha=0.5)
#
# ax.legend(["SGD","Pranam-SGD","Adam","Pranam-Adam","RMSProp","Pranam-RMSProp","AdaDelta","Pranam-AdaDelta"])#,prop={'size': 12})
# plt.savefig(os.path.join(out_path,  "optimizers.png"))
# plt.close()