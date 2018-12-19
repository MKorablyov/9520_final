import numpy as np
import tensorflow as tf
import torch




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


def schwefel_torch(x, xmin=-1, xmax=1):
    """
    https://www.sfu.ca/~ssurjano/schwef.html
    """
    x = rescale(x, xmin, xmax, -500, 500)
    print "scaled X:", x
    result = x.abs().sqrt().sin().mul(x).sum(1).mul(-1).add(418.9829 * x.size(1))
    return result



x = np.array([[0,0],[-10.064, 0.],[-0.,-0.007],[-0.001, -0.],[-14.845,0.],[6.843,0.001],
              [6.843,0.],[6.843,0.001],[3.093,-0.585]],dtype=np.float32)
loss = schwefel(x)
sess = tf.Session()

stuff = [1.0,2,3,4]

print (torch.tensor(stuff)).tanh()
print sess.run(tf.nn.tanh(tf.convert_to_tensor(stuff)))



#print sess.run([loss])
#print schwefel_torch(torch.tensor(x))