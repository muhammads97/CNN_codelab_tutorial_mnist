from neural_network import init
from neural_network import layers
import tensorflow as tf

def five_layers_conv_model(X, keep):
    layer = layers.conv_layer(X, 6, 6, 1, 6, 1, tf.nn.relu)
    layer = layers.conv_layer(layer, 5, 5, 6, 12, 2, tf.nn.relu)
    layer = layers.conv_layer(layer, 4, 4, 12, 24, 2, tf.nn.relu)
    layer = tf.reshape(layer, [-1, 7 * 7 * 24])
    layer = layers.dropout_FC_layer(layer, 7 * 7 * 24, 200, keep, tf.nn.relu)
    layer = layers.FC_layer(layer, 200, 10, tf.identity)
    return layer

def five_layers_FC_model(X, keep):
    layer = tf.reshape(X, [-1, 784])
    layer = layers.FC_layer(layer, 784, 200, tf.nn.relu)
    layer = layers.dropout_layer(layer, keep)
    layer = layers.FC_layer(layer, 200, 100, tf.nn.relu)
    layer = layers.dropout_layer(layer, keep)
    layer = layers.FC_layer(layer, 100, 60, tf.nn.relu)
    layer = layers.dropout_layer(layer, keep)
    layer = layers.FC_layer(layer, 60, 30, tf.nn.relu)
    layer = layers.dropout_layer(layer, keep)
    layer = layers.FC_layer(layer, 30, 10, tf.identity)
    return layer
