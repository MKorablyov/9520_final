# I want to show what happens when I am fitting one wavelet
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# t = np.linspace(-20,20,200)
# sig = np.sin(2* t) / t
# plt.plot(sig)
# plt.show()
# import mlpy.wavelet as wave
# x = np.random.sample(512)
# scales = wave.autoscales(N=x.shape[0], dt=1, dj=0.25, wf='dog', p=2)
# X = wave.cwt(x=x, dt=1, scales=scales, wf='dog', p=2)
# fig = plt.figure(1)
# ax1 = plt.subplot(2,1,1)
# p1 = ax1.plot(x)
# ax1.autoscale_view(tight=True)
# ax2 = plt.subplot(2,1,2)
# p2 = ax2.imshow(np.abs(X), interpolation='nearest')
# plt.show()


from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt


fig = plt.figure()
fig.set_size_inches(12.8, 12.8)
ax = fig.gca(projection='3d')
# Grab some test data.
#X, Y, Z = axes3d.get_test_data(0.05)

# Plot a basic wireframe.
#ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10)

x = np.linspace(-1, 1, 100)
y = np.linspace(-1, 1, 100)
xv, yv = np.meshgrid(x, y)
surf = ax.plot_surface(xv,yv, xv**2 + yv**2,rstride=1, cstride=1, cmap=cm.coolwarm, color='c', alpha=0.3, linewidth=0)

x = np.random.uniform(-1,1,size=10)
y = np.random.uniform(-1,1,size=10)
ax.scatter(x,y,x**2+y**2,color="k",s=20)
#surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)


plt.savefig("/home/maksym/Desktop/test.png")
#plt.show()