import tensorflow as tf
import math

def get_placeholders(input_shape, output_shape, lr_decay = False, dropout = False):
    x     = tf.placeholder(tf.float32, input_shape)
    Y     = tf.placeholder(tf.float32, output_shape)
    if lr_decay and dropout:
        step  = tf.placeholder(tf.float32)
        keep = tf.placeholder(tf.float32)
        return x, Y, step, keep 
    elif lr_decay:
        step  = tf.placeholder(tf.float32)
        return x, Y, step
    elif dropout:
        keep = tf.placeholder(tf.float32)
        return x, Y, keep
    else:
        return x, Y, step, keep

def get_Wb(shape, random_type = "normal"):
    if random_type == "normal":
        w = tf.Variable(tf.truncated_normal(shape, stddev = 0.1))
        b = tf.Variable(tf.ones([shape[-1]])/10)
        return w, b

def get_lr(start, step = False):
    if not step:
        return start
    else:
        return 0.0001 + tf.train.exponential_decay(start, step, 2000, 1/math.e)