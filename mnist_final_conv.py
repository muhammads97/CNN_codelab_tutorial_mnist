import tensorflow as tf
import tensorflowvisu
import mnistdata
import math
from neural_network import init 
from neural_network import model
from neural_network import optimizer as opt
print("Tensorflow version " + tf.__version__)
tf.set_random_seed(0)

# ...............................reading data.................................
mnist = mnistdata.read_data_sets("data", one_hot=True, reshape=False)

# ...............................initialization...............................

inputs, labels, step, keep = init.get_placeholders([None, 28, 28, 1], [None, 10], True, True)
lr = init.get_lr(0.003, True)

# ....................................model...................................

logits = model.five_layers_conv_model(inputs, keep)

# .................................train step.................................

cross_entropy, accuracy, train_step = opt.train_step(lr, logits, labels, 100, tf.train.AdamOptimizer)

# .................................visualization..............................

I = tensorflowvisu.tf_format_mnist_images(inputs, tf.nn.softmax(logits), labels)
It = tensorflowvisu.tf_format_mnist_images(inputs, tf.nn.softmax(logits), labels, 1000, lines=25)
datavis = tensorflowvisu.MnistDataVis()

# init
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

def training_step(i, update_test_data, update_train_data):
    batch_X, batch_Y = mnist.train.next_batch(100)

    if update_train_data:
        a, c, im, l = sess.run([accuracy, cross_entropy, I, lr],
                                     feed_dict={inputs: batch_X, labels: batch_Y, step: i, keep:1.0})
        print(str(i) + ": accuracy:" + str(a) + " loss: " + str(c) + " (lr:" + str(l) + ")")
        datavis.append_training_curves_data(i, a, c)
        datavis.update_image1(im)

    if update_test_data:
        a, c, im = sess.run([accuracy, cross_entropy, It], feed_dict={inputs: mnist.test.images, labels: mnist.test.labels, keep:1.0})
        print(str(i) + ": ********* epoch " + str(i*100//mnist.train.images.shape[0]+1) + " ********* test accuracy:" + str(a) + " test loss: " + str(c))
        datavis.append_test_curves_data(i, a, c)
        datavis.update_image2(im)

    sess.run(train_step, feed_dict={inputs: batch_X, labels: batch_Y, step: i, keep:0.75})

for i in range(10000+1): 
    training_step(i, i % 50 == 0, i % 10 == 0)
print("max test accuracy: " + str(datavis.get_max_test_accuracy()))