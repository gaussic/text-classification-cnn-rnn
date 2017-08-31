#!/usr/bin/python
# -*- coding: utf-8 -*-

from rnn_model import *
from cnn_model import *
from configuration import *

def run_epoch(cnn=True):
    # 载入数据
    print('Loading data...')
    start_time = time.time()

    x_train, y_train, x_test, y_test, x_val, y_val, words = preocess_file()

    if cnn:
        print('Using CNN model...')
        config = TCNNConfig()
        config.vocab_size = len(words)
        model = TextCNN(config)
        tensorboard_dir = 'tensorboard/textcnn'
    else:
        print('Using RNN model...')
        config = TRNNConfig()
        config.vocab_size = len(words)
        model = TextRNN(config)
        tensorboard_dir = 'tensorboard/textrnn'

    end_time = time.time()
    time_dif = end_time - start_time
    time_dif = timedelta(seconds=int(round(time_dif)))
    print('Time usage:', time_dif)

    print('Constructing TensorFlow Graph...')
    session = tf.Session()
    session.run(tf.global_variables_initializer())

    # 配置 tensorboard
    tf.summary.scalar("loss", model.loss)
    tf.summary.scalar("accuracy", model.acc)

    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)

    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tensorboard_dir)
    writer.add_graph(session.graph)

    # 生成批次数据
    print('Generating batch...')
    batch_train = batch_iter(list(zip(x_train, y_train)),
        config.batch_size, config.num_epochs)

    def feed_data(batch):
        """准备需要喂入模型的数据"""
        x_batch, y_batch = zip(*batch)
        feed_dict = {
            model.input_x: x_batch,
            model.input_y: y_batch
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
            loss, acc = session.run([model.loss, model.acc],
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
            loss_train, acc_train = session.run([model.loss, model.acc],
                feed_dict=feed_dict)
            loss, acc = evaluate(x_val, y_val)

            # 时间
            end_time = time.time()
            time_dif = end_time - start_time
            time_dif = timedelta(seconds=int(round(time_dif)))

            msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%},'\
                + ' Val Loss: {3:>6.2}, Val Acc: {4:>7.2%}, Time: {5}'
            print(msg.format(i + 1, loss_train, acc_train, loss, acc, time_dif))

        session.run(model.optim, feed_dict=feed_dict)  # 运行优化

    # 最后在测试集上进行评估
    print('Evaluating on test set...')
    loss_test, acc_test = evaluate(x_test, y_test)
    msg = 'Test Loss: {0:>6.2}, Test Acc: {1:>7.2%}'
    print(msg.format(loss_test, acc_test))

    session.close()

if __name__ == '__main__':
    run_epoch(cnn=True)
