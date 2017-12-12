import dataset
import tensorflow as tf
import time
from datetime import timedelta
import math
import random
import numpy as np

# Adding Seed so that random initialization is consistent
from numpy.random import seed

seed(1)
from tensorflow import set_random_seed

set_random_seed(2)

# Prepare input data
classes = ['dogs', 'cats']
num_classes = len(classes)

# all data is for train
validation_size = 0
img_size = 128
num_channels = 3
test_path = 'testing_data'


def load_test_dataset():
    return dataset.read_train_sets(test_path, img_size, classes, validation_size=validation_size)


def load_graph(sess):
    saver = tf.train.import_meta_graph(
        '/home/igor/university/machine-learning/cv-tricks.com/Tensorflow-tutorials/tutorial-2-image-classifier/models/dogs-cats-model.meta')

    saver.restore(sess, tf.train.latest_checkpoint('./'))
    return tf.get_default_graph()


def test_feed(graph, data):
    x = graph.get_tensor_by_name("x:0")
    y_true = graph.get_tensor_by_name("y_true:0")
    y_pred = graph.get_tensor_by_name("y_pred:0")

    ### Creating the feed_dict that is required to be fed to calculate y_pred
    x_batch, y_true_batch, _, cls_batch = data.train.next_batch(len(data.train.images))
    # x_valid_batch, y_valid_batch, _, valid_cls_batch = data.valid.next_batch(batch_size)

    feed_dict_tr = {x: x_batch,
                    y_true: y_true_batch}
    return (y_pred, feed_dict_tr)


def get_acc(graph, sess, y_pred, feed_dict_tr):
    result = sess.run(y_pred, feed_dict=feed_dict_tr)
    final_res = np.array([[1 if x1 > x2 else 0, 1 if x2 > x1 else 0] for x1, x2 in result])
    y_true_batch = feed_dict_tr.get(graph.get_tensor_by_name("y_true:0"))
    correct_prediction = tf.equal(final_res, y_true_batch)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    acc = sess.run(accuracy, feed_dict=feed_dict_tr)
    return acc

data = load_test_dataset()

sess = tf.Session()
graph = load_graph(sess)

(y_pred, feed_dict_tr) = test_feed(graph, data)
acc = get_acc(graph, sess, y_pred, feed_dict_tr)
print('acc: ', acc)
