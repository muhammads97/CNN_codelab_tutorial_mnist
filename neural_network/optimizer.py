import tensorflow as tf

def train_step(lr, logits, labels, batch_size = 100, opt = tf.train.AdamOptimizer):
    cee   = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    cee   = tf.reduce_mean(cee) * batch_size
    Y     = tf.nn.softmax(logits)
    preds = tf.equal(tf.argmax(Y, 1), tf.argmax(labels, 1))
    acc   = tf.reduce_mean(tf.cast(preds, tf.float32))
    opt   = opt(lr).minimize(cee)
    return cee, acc, opt
