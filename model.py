#!/usr/bin/python
# -*- coding: utf-8 -*-

from data.cnews_loader import *
import tensorflow as tf


class TCNNConfig(object):
    """部分配置参数"""

    # 模型参数
    embedding_dim = 64
    seq_length = 600
    num_classes = 10
    num_filters = 256
    kernel_size = 5
    vocab_size = 5000

    hidden_dim = 128

    dropout_keep_prob = 0.8
    learning_rate = 0.05

    batch_size = 64


class TextCNN(object):
    """文本分类，CNN模型"""
    def __init__(self, config):
        self.config = config

        self.input_x = tf.placeholder(tf.int32,
            [None, self.config.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32,
            [None, self.config.num_classes], name='input_y')

        self.cnn()

    def input_embedding(self):
        with tf.device('/cpu:0'):
            embedding = tf.get_variable('embedding',
                [self.config.vocab_size, self.config.embedding_dim])
            _inputs = tf.nn.embedding_lookup(embedding, self.input_x)
        return _inputs

    def cnn(self):
        embedding_inputs = self.input_embedding()

        with tf.name_scope("cnn"):
            conv = tf.layers.conv1d(embedding_inputs,
                self.config.num_filters,
                self.config.kernel_size, name='conv')

            # global max pooling
            gmp = tf.reduce_max(conv, reduction_indices=[1])

        # 全连接层，后面接dropout以及relu激活
        with tf.name_scope("score"):
            fc = tf.layers.dense(gmp, self.config.hidden_dim, name='fc1')
            fc = tf.contrib.layers.dropout(fc, self.config.dropout_keep_prob)
            fc = tf.nn.relu(fc)

            self.logits = tf.layers.dense(fc, self.config.num_classes, name='fc2')
            self.pred_y = tf.nn.softmax(self.logits)

        with tf.name_scope("loss"):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                logits=self.logits, labels=self.input_y)
            loss = tf.reduce_mean(cross_entropy)
            self.loss = loss

        with tf.name_scope("optimize"):
            optimizer = tf.train.AdamOptimizer(
                learning_rate=self.config.learning_rate)
            self.optim = optimizer.minimize(self.loss)

        with tf.name_scope("accuracy"):
            correct_pred = tf.equal(tf.argmax(self.input_y, 1),
                tf.argmax(self.pred_y, 1))
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))



def run_graph():
    config = TCNNConfig()
    tcnn = TextCNN(config)
    session = tf.Session()
    session.run(tf.global_variables_initializer())

    tensorboard_dir = 'tensorboard/textcnn'
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)

    writer = tf.summary.FileWriter(tensorboard_dir)
    writer.add_graph(session.graph)




def run_epoch():
    config = TCNNConfig()
    tcnn = TextCNN(config)

    session = tf.Session()
    session.run(tf.global_variables_initializer())

    x_train, y_train, x_test, y_test, x_val, y_val = preocess_file()

    batch_train = batch_iter(list(zip(x_train, y_train)), config.batch_size)

    i = 0
    for batch in batch_train:
        x_batch, y_batch = zip(*batch)

        feed_dict_train = {
            tcnn.input_x: x_batch,
            tcnn.input_y: y_batch
        }

        session.run(tcnn.optim, feed_dict=feed_dict_train)

        if i % 10 == 0:
            loss, error = session.run([tcnn.loss, tcnn.error],
                feed_dict=feed_dict_train)

            msg = 'Iter: {0:>5}, Loss: {1:>3.3}, Error: {2:>3.3}'
            print(msg.format(i + 1, loss, error))
        i += 1




if __name__ == '__main__':
    run_epoch()
    # run_graph()
