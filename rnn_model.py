#!/usr/bin/python
# -*- coding: utf-8 -*-

from data.cnews_loader import *
import tensorflow as tf

import time
from datetime import timedelta

import shutil

class TRNNConfig(object):
    """配置参数"""

    # 模型参数
    embedding_dim = 64      # 词向量维度
    seq_length = 600        # 序列长度
    num_classes = 10        # 类别数
    vocab_size = 5000       # 词汇表达小

    num_layers= 2           # 隐藏层层数
    hidden_dim = 128        # 隐藏层神经元
    rnn = 'gru'             # lstm 或 gru

    dropout_keep_prob = 0.8 # dropout保留比例
    learning_rate = 1e-3    # 学习率

    batch_size = 128         # 每批训练大小
    num_epochs = 10          # 总迭代轮次


class TextRNN(object):
    """文本分类，RNN模型"""
    def __init__(self, config):
        self.config = config

        self.input_x = tf.placeholder(tf.int32,
            [None, self.config.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32,
            [None, self.config.num_classes], name='input_y')

        self.rnn()

    def input_embedding(self):
        """词嵌入"""
        with tf.device('/cpu:0'):
            embedding = tf.get_variable('embedding',
                [self.config.vocab_size, self.config.embedding_dim])
            _inputs = tf.nn.embedding_lookup(embedding, self.input_x)
        return _inputs

    def rnn(self):
        """rnn模型"""

        def lstm_cell():
            return tf.contrib.rnn.BasicLSTMCell(self.config.hidden_dim,
                state_is_tuple=True)

        def gru_cell():
            return tf.contrib.rnn.GRUCell(self.config.hidden_dim)

        def dropout():
            if (self.config.rnn == 'lstm'):
                cell = lstm_cell()
            else:
                cell = gru_cell()

            return tf.contrib.rnn.DropoutWrapper(cell,
                output_keep_prob=self.config.dropout_keep_prob)

        embedding_inputs = self.input_embedding()

        with tf.name_scope("rnn"):
            cells = [dropout() for _ in range(self.config.num_layers)]
            rnn_cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)

            _outputs, _ = tf.nn.dynamic_rnn(cell=rnn_cell,
                inputs=embedding_inputs, dtype=tf.float32)
            last = _outputs[:, -1, :]

        with tf.name_scope("score"):
            # 全连接层，后面接dropout以及relu激活
            fc = tf.layers.dense(last, self.config.hidden_dim, name='fc1')
            fc = tf.contrib.layers.dropout(fc,
                self.config.dropout_keep_prob)
            fc = tf.nn.relu(fc)

            # 分类器
            self.logits = tf.layers.dense(fc, self.config.num_classes,
                name='fc2')
            self.pred_y = tf.nn.softmax(self.logits)

        with tf.name_scope("loss"):
            # 损失函数，交叉熵
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)

        with tf.name_scope("optimize"):
            # 优化器
            optimizer = tf.train.AdamOptimizer(
                learning_rate=self.config.learning_rate)
            self.optim = optimizer.minimize(self.loss)

        with tf.name_scope("accuracy"):
            # 准确率
            correct_pred = tf.equal(tf.argmax(self.input_y, 1),
                tf.argmax(self.pred_y, 1))
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

def run_epoch():

    # 载入数据
    print('Loading data...')
    start_time = time.time()
    config = TRNNConfig()
    x_train, y_train, x_test, y_test, \
        x_val, y_val, words = preocess_file()
    config.vocab_size = len(words)

    end_time = time.time()
    time_dif = end_time - start_time
    time_dif = timedelta(seconds=int(round(time_dif)))
    print('Time usage:', time_dif)

    print('Constructing Model...')
    tcnn = TextRNN(config)
    session = tf.Session()
    session.run(tf.global_variables_initializer())

    # 配置 tensorboard
    tf.summary.scalar("loss", tcnn.loss)
    tf.summary.scalar("accuracy", tcnn.acc)

    tensorboard_dir = 'tensorboard/textcnn'
    if os.path.exists(tensorboard_dir):
        shutil.rmtree(tensorboard_dir)
        os.makedirs(tensorboard_dir)

    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tensorboard_dir)
    writer.add_graph(session.graph)

    # 生成批次数据
    batch_train = batch_iter(list(zip(x_train, y_train)),
        config.batch_size, config.num_epochs)

    def feed_data(batch):
        x_batch, y_batch = zip(*batch)
        feed_dict = {
            tcnn.input_x: x_batch,
            tcnn.input_y: y_batch
        }
        return feed_dict, len(x_batch)

    def evaluate(x_, y_):
        """
        模型评估
        一次运行所有的数据会OOM，所以需要分批和汇总
        """
        batch_eval = batch_iter(list(zip(x_, y_)), 128, 1)

        total_loss = 0.0
        total_acc = 0.0
        cnt = 0
        for batch in batch_eval:
            feed_dict, cur_batch_len = feed_data(batch)
            loss, acc = session.run([tcnn.loss, tcnn.acc],
                feed_dict=feed_dict)
            total_loss += loss * cur_batch_len
            total_acc += acc * cur_batch_len
            cnt += cur_batch_len

        return total_loss / cnt, total_acc / cnt

    # 训练与验证
    print('Training and evaluating...')
    start_time = time.time()
    for i, batch in enumerate(batch_train):
        feed_dict, _ = feed_data(batch)

        if i % 5 == 0:  # 每5次将训练结果写入tensorboard scalar
            s = session.run(merged_summary, feed_dict=feed_dict)
            writer.add_summary(s, i)

        if i % 200 == 0:  # 每200次输出在训练集和验证集上的性能
            loss_train, acc_train = session.run([tcnn.loss, tcnn.acc],
                feed_dict=feed_dict)
            loss, acc = evaluate(x_val, y_val)

            # 时间
            end_time = time.time()
            time_dif = end_time - start_time
            time_dif = timedelta(seconds=int(round(time_dif)))

            msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%},'\
                + ' Val Loss: {3:>6.2}, Val Acc: {4:>7.2%}, Time: {5}'
            print(msg.format(i + 1, loss_train, acc_train, loss, acc, time_dif))

        session.run(tcnn.optim, feed_dict=feed_dict)  # 运行优化

    # 最后在测试集上进行评估
    print('Evaluating on test set...')
    loss_test, acc_test = evaluate(x_test, y_test)
    msg = 'Test Loss: {0:>6.2}, Test Acc: {1:>7.2%}'
    print(msg.format(loss_test, acc_test))

    session.close()

if __name__ == '__main__':
    run_epoch()
