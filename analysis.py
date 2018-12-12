import tensorflow as tf
import os
import numpy as np
from dataprep import *
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import datetime



print "Loading data now..."

#Filter variables
sr = 2000.0
lowcut = 20  # 50 to 200 is good aswell
highcut = 300.0

X_val = np.load('/home/zain/Software/Stethoscope/ClassificationML/data/RawData/Rawvaldata.npy')

for i in range(len(X_val)):
    X_val[i] = bandpass(X_val[i], lowcut, highcut, sr, order=6)
    X_val[i] = normalizedata(X_val[i])


#THis will down sample to 200Hz form 2000Hz
X_val = X_val[:].reshape(len(X_val),-1,10).mean(axis=2)
#plot(X_val[0])
#plot(X_val[0].reshape(-1, 10).mean(axis=1))
#X_val = X_val.reshape((len(X_val),400,1))

Y_val = np.load('/home/zain/Software/Stethoscope/ClassificationML/data/RawData/Rawvallabel.npy')
Y_val = one_hot(Y_val)

zeroData, X_val, zeroLabel, Y_val = train_test_split(X_val, Y_val, test_size=0.99999, random_state=40)


X_test = np.load('/home/zain/Software/Stethoscope/ClassificationML/data/RawData/Rawtestdata.npy')

for i in range(len(X_test)):
    X_test[i] = bandpass(X_test[i], lowcut, highcut, sr, order=6)
    X_test[i] = normalizedata(X_test[i])

#plot(X_test[0].reshape(-1, 10).mean(axis=1))
#THis will down sample to 200Hz form 2000Hz
X_test = X_test.reshape(len(X_test),-1,10).mean(axis=2)
#plot(X_test[0])
#X_test = X_test.reshape((len(X_test),400,1))

Y_test = np.load('/home/zain/Software/Stethoscope/ClassificationML/data/RawData/Rawtestlabel.npy')
Y_test = one_hot(Y_test)

zeroData, X_test, zeroLabel, Y_test = train_test_split(X_test, Y_test, test_size=0.99999, random_state=42)

X_test = X_test.reshape((len(X_test),400,1))

X_val = X_val.reshape((len(X_val),400,1))

print "Data has been loaded."


batch_size = 200
seq_len = 400
learning_rate = 0.001
epochs = 600
n_classes = 2
n_channels = 1

graph = tf.Graph()


def train( dataToClassify, dataLabel):

    with graph.as_default():
        inputs_ = tf.placeholder(tf.float32,[None, seq_len, n_channels], name='inputs')
        labels_ = tf.placeholder(tf.float32, [None, n_classes], name = 'labels')
        keep_prob_ = tf.placeholder(tf.float32,name='keep')
        learning_rate_ = tf.placeholder(tf.float32,name='learning_rate')

    with graph.as_default():
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

    with graph.as_default():
            flat = tf.reshape(max_pool_4, (-1, max_pool_4.shape[1]*max_pool_4.shape[2]))
            flat = tf.nn.dropout(flat, keep_prob=keep_prob_)

            logits = tf.layers.dense(flat, n_classes)

            loss_func = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels_)
            cost = tf.reduce_mean(loss_func)
            optimizer = tf.train.AdamOptimizer(learning_rate_, beta1=0.4).minimize(cost)

            correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(labels_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

            tf.summary.scalar(name = 'Loss Function', tensor=cost)
            tf.summary.scalar(name = 'Training Accuracy', tensor=accuracy)
            tf.summary.histogram(name='Classifier Output Distribution', values=logits)
            summary_op = tf.summary.merge_all()

    with graph.as_default():
        saver = tf.train.Saver()

    with tf.Session(graph=graph) as sess:
        sess.run(tf.global_variables_initializer())

        #This will load up a saved model which will simply Classify the given data
        results_path = '/home/zain/PycharmProjects/StethPy/Results/2018-06-21 02:49:30.196622_LR:0.001_Epochs:600_1D-CNN/SavedModel/'
        tstart = datetime.datetime.now()
        saver.restore(sess, save_path=tf.train.latest_checkpoint(results_path))

        feed = {inputs_: dataToClassify, labels_: dataLabel, keep_prob_: 1.0}

        prediction, acc = sess.run([logits, accuracy], feed_dict=feed)
        prediction = (prediction>0).astype(float)
        tend = datetime.datetime.now()
        print 'Input Class Label is:', dataLabel
        print 'Predicted Class:', prediction, 'Classification Accuracy:', acc
        print 'Feedforeward Time in microseconds:', (tend-tstart).microseconds

if __name__ == '__main__':
    train(X_test[7].reshape(1,400,1),Y_test[7].reshape(1,2))
