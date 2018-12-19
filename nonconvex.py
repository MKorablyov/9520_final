import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import os,time


def rescale(x, a, b, c, d):
    """
    Rescales variable from [a, b] to [c, d]
    """
    return c + ((d - c) / (b - a)) * (x - a)

def ackley(x, xmin=-1, xmax=1):
    """
    https://www.sfu.ca/~ssurjano/ackley.html
    """
    a = 20.0
    b = 0.2
    c = 2 * np.pi
    x = rescale(x, xmin, xmax, -32.768, 32.768)
    term1 = -a * tf.exp(-b * tf.reduce_mean(x**2,axis=1)**0.5)
    term2 = -tf.exp(tf.reduce_mean(tf.cos(c*x),axis=1))
    return term1 + term2 + a + tf.exp(1.0)


def schwefel(x, xmin=-1, xmax=1):
    """
    https://www.sfu.ca/~ssurjano/schwef.html
    """
    x = rescale(x, xmin, xmax, -500, 500)
    #result = x.abs().sqrt().sin().mul(x).sum(1).mul(-1).add(418.9829 * x.size(1))
    result = 418.9829 * tf.to_float(tf.shape(x)[1]) - tf.reduce_sum(tf.sin(tf.abs(x)**0.5) * x,axis=1)
    return result


def perceptron(sizes,sess):
    # init network (first 2 layers)
    # fixme changed the original implementation
    top_layer = tf.get_variable("embed",
                                shape=[sizes[0],sizes[1]],
                                initializer=tf.contrib.layers.xavier_initializer())
#    input = tf.range(0,sizes[0])
#    embed_params = tf.get_variable("perc_embed",
#                                   shape=[sizes[0],sizes[1]],
#                                   initializer=tf.contrib.layers.xavier_initializer())
#    top_layer = tf.nn.embedding_lookup(params=embed_params,ids=input)
    # build network
    for i in range(1, len(sizes) - 1):
        name = "perceptron_fc" + str(i)
        shape = [sizes[i],sizes[i+1]]
        w = tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())
        top_layer = tf.nn.relu(tf.matmul(top_layer,w))
    # rescale with tg activation
    #return top_layer
    sess.run(tf.global_variables_initializer()) # FIXME: I need to initialize each of the weights !
    scaling = 1.7159 / tf.maximum(tf.abs(tf.reduce_max(top_layer)), tf.abs(tf.reduce_min(top_layer)))
    scaling_const = sess.run(scaling)
    out = tf.nn.tanh(scaling_const * top_layer)
    return out

sess = tf.Session()
perc_embed = perceptron([250,2],sess)
#perc_embed = perceptron([6400,32,1024])
#print perc_embed

losses = schwefel(perc_embed)
train_step = tf.train.GradientDescentOptimizer(learning_rate=0.000003).minimize(losses)
sess.run(tf.global_variables_initializer())
test_loss = tf.placeholder(tf.float32,shape=[None,2])
test_scwefel = schwefel(test_loss)


# Ideas of how to build a network
# I have 6400 functions of size 1024, each one results in a loss

# in my network I have 1024 weights, each one results in a loss
# in the layer before top, I can get 6400 things, and use that to update

# in the layer 2 before top, I can use the lowest value from the layer above
# but this would partially defeat the purpose (all points of the layer above would come together)
# I could allow 6,400 x 1024


np.set_printoptions(precision=3,suppress=True)
for i in range(10000):
    _perc_embed,_losses,_ = sess.run([perc_embed, losses, train_step])
    #for i in range(1000):
    #_perc_embed, _losses = sess.run([perc_embed, losses])
    print "perc embed", _perc_embed
    print "losses", _losses
    #    print "min loss", np.argmin(_losses), _losses[np.argmin(_losses)]
    #_min_embed = _perc_embed[np.argmin(_losses[0]),:]
    #    print "min embed", _min_embed
    #    #print _losses.shape, perc_embed.shape
    print "step:", i, "mean loss", np.mean(_losses), "min loss", np.min(_losses)
    # print "perc embed:", np.around(_perc_embed,decimals=2)
    # print sess.run([p])[0]
    #_test_vals = np.expand_dims(_losses[losses.argmin()],0)
    # print "test vals", _test_vals
    # print "test scwefel:", sess.run([test_loss, test_scwefel],feed_dict={test_loss:_perc_embed})
    # print "test scwefel:", sess.run([test_loss, test_scwefel], feed_dict={test_loss: _test_vals})
    # print "test scwefel:", sess.run([test_loss, test_scwefel], feed_dict={test_loss: _test_vals})
    print "smallest loss;", _losses[_losses.argmin()],"at:", _perc_embed[_losses.argmin()]


fig = plt.figure()
fig.set_size_inches(12.8, 12.8)
ax = fig.gca(projection='3d')
x = np.linspace(-1, 1, 100)
y = np.linspace(-1, 1, 100)
xv, yv = np.asarray(np.meshgrid(x, y),dtype=np.float32)
zv = schwefel(np.stack([xv,yv],axis=1))
zv, = sess.run([zv])
surf = ax.plot_surface(xv, yv, zv, rstride=1, cstride=1, cmap=cm.coolwarm, color='c', alpha=0.3, linewidth=0)
#plt.show()
#xys = network().detach().numpy()
#zs = the_function(network()).detach().numpy()
#print zs
#ax.scatter(xys[:, 0], xys[:, 1], zs, color="k", s=20)
#if not os.path.exists(os.path.join(args.save_dir, str(lr))): os.makedirs(os.path.join(args.save_dir, str(lr)))
#plt.show()
plt.savefig(os.path.join("/home/maksym/Desktop/SLT/plots", "surf_" + ".png"))
plt.close()

#np.random.uniform()