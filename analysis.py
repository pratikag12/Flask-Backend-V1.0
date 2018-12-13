import tensorflow as tf
import os
import numpy as np
import datetime


class CNNModel():

    def __init__(self):
        self.graph = tf.Graph()

    def runanalysis(self, path):
        return self.classify(self.loadnpy(path))

    def loadnpy(self, pathToNpy):
        return np.load(pathToNpy)

    def classify(self, dataToClassify):
        HeartDATA = dataToClassify[:].reshape(1, -1, 10).mean(axis=2)
        HeartDATA = HeartDATA.reshape(1, 400, 1)

        seq_len = 400
        n_classes = 2
        n_channels = 1

        with self.graph.as_default():
            inputs_ = tf.placeholder(tf.float32,[None, seq_len, n_channels], name='inputs')
            labels_ = tf.placeholder(tf.float32, [None, n_classes], name = 'labels')
            keep_prob_ = tf.placeholder(tf.float32,name='keep')
            learning_rate_ = tf.placeholder(tf.float32,name='learning_rate')

        with self.graph.as_default():
            conv1 = tf.layers.conv1d(inputs=inputs_, kernel_initializer=tf.random_normal_initializer(stddev = 0.5), filters=18, kernel_size=2, strides=1,
                                     padding='same', activation=tf.nn.relu)
            max_pool_1 = tf.layers.max_pooling1d(inputs=conv1, pool_size=2, strides=2, padding='same')

            conv2 = tf.layers.conv1d(inputs=max_pool_1, kernel_initializer=tf.random_normal_initializer(stddev = 0.5), filters=36, kernel_size=2, strides=1,
                                     padding='same', activation=tf.nn.relu)
            max_pool_2 = tf.layers.max_pooling1d(inputs=conv2, pool_size=2, strides=2, padding='same')

            conv3 = tf.layers.conv1d(inputs=max_pool_2, kernel_initializer=tf.random_normal_initializer(stddev = 0.5), filters=72, kernel_size=2, strides=1,
                                     padding='same', activation=tf.nn.relu)
            max_pool_3 = tf.layers.max_pooling1d(inputs=conv3, pool_size=2, strides=2, padding='same')

            conv4 = tf.layers.conv1d(inputs=max_pool_3, kernel_initializer=tf.random_normal_initializer(stddev = 0.5), filters=144, kernel_size=2, strides=1,
                                     padding='same', activation=tf.nn.relu)
            max_pool_4 = tf.layers.max_pooling1d(inputs=conv4, pool_size=2, strides=2, padding='same')

        with self.graph.as_default():
                flat = tf.reshape(max_pool_4, (-1, max_pool_4.shape[1]*max_pool_4.shape[2]))
                flat = tf.nn.dropout(flat, keep_prob=keep_prob_)

                logits = tf.layers.dense(flat, n_classes)

                loss_func = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels_)
                cost = tf.reduce_mean(loss_func)
                optimizer = tf.train.AdamOptimizer(learning_rate_, beta1=0.4).minimize(cost)

                correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(labels_, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

        with self.graph.as_default():
            saver = tf.train.Saver()

        with tf.Session(graph=self.graph) as sess:
            sess.run(tf.global_variables_initializer())
            #This will load up a saved model which will simply Classify the given data
            results_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'ml') #Get trained data info
            tstart = datetime.datetime.now()
            saver.restore(sess, save_path=tf.train.latest_checkpoint(results_path))
            labelPH = [[0., 1.]]
            labelPHn = np.float32(labelPH)

            feed = {inputs_: HeartDATA, labels_: labelPHn, keep_prob_: 1.0}

            prediction, acc = sess.run([logits, accuracy], feed_dict=feed)
            prediction = (prediction>0).astype(float)
            tend = datetime.datetime.now()
            return (prediction, (tend-tstart).microseconds)