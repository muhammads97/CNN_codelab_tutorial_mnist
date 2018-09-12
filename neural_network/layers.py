import tensorflow as tf
from neural_network import init 

def FC_layer(X, input_size, output_size, activation = tf.nn.relu):
    weights, biases = init.get_Wb([input_size, output_size])
    layer = tf.matmul(X, weights) + biases
    layer = activation(layer)
    return layer

def dropout_layer(X, keep):
    return tf.nn.dropout(X, keep)

def dropout_FC_layer(X, input_size, output_size, keep, activation = tf.nn.relu):
    layer = FC_layer(X, input_size, output_size, activation)
    layer = dropout_layer(layer, keep)
    return layer

def conv_layer(X, h, w, c, filters, stride = 1, activation = tf.nn.relu):
    weights, biases = init.get_Wb([h, w, c, filters])
    layer = tf.nn.conv2d(X, weights, strides=[1, stride, stride, 1], padding='SAME')
    layer = activation(layer + biases)
    return layer