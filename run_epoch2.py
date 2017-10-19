#!/usr/bin/python
# -*- coding: utf-8 -*-

from rnn_model import *
from cnn_model import *
from configuration import *
from data.cnews_loader import *
from sklearn import metrics

import time
from datetime import timedelta

def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

def construct_model(vocab_size, cnn=True, training=True):
    """构建模型"""
    tensorboard_dir = ''
    if cnn:
        print('Using CNN model...')
        config = TCNNConfig()
        config.vocab_size = vocab_size
        model = TextCNN(config)
        save_dir = 'checkpoints/textcnn'
        if training:
            tensorboard_dir = 'tensorboard/textcnn'
    else:
        print('Using RNN model...')
        config = TRNNConfig()
        config.vocab_size = vocab_size
        model = TextRNN(config)
        save_dir = 'checkpoints/textrnn'
        if training:
            tensorboard_dir = 'tensorboard/textrnn'

    return model, save_dir, tensorboard_dir

def run_epoch(cnn=True):
    # 载入数据
    print('Loading data...')
    start_time = time.time()

    if not os.path.exists('data/cnews/vocab_cnews.txt'):
        build_vocab('data/cnews/cnews.train.txt')

    x_train, y_train, x_test, y_test, x_val, y_val, words = preocess_file()

    model, save_dir, tensorboard_dir = construct_model(len(words))

    time_dif = get_time_dif(start_time)
    print('Time usage:', time_dif)

    # 配置模型保存路径
    saver = tf.train.Saver()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, 'best_validation')   # 最佳验证结果保存路径

    # 配置 tensorboard
    tf.summary.scalar("loss", model.loss)
    tf.summary.scalar("accuracy", model.acc)

    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)

    print('Constructing TensorFlow Graph...')
    session = tf.Session()
    session.run(tf.global_variables_initializer())

    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tensorboard_dir)
    writer.add_graph(session.graph)

    def feed_data(batch):
        """准备需要喂入模型的数据"""
        x_batch, y_batch = zip(*batch)
        feed_dict = {
            model.input_x: x_batch,
            model.input_y: y_batch
        }
        return feed_dict, len(x_batch)

    def evaluate(x_, y_):
        """模型评估，一次运行所有的数据会OOM，所以需要分批和汇总"""
        batch_eval = batch_iter(list(zip(x_, y_)), 128, 1)

        total_loss = 0.0
        total_acc = 0.0
        for batch in batch_eval:
            feed_dict, cur_batch_len = feed_data(batch)
            feed_dict[model.keep_prob] = 1.0
            loss, acc = session.run([model.loss, model.acc], feed_dict=feed_dict)
            total_loss += loss * cur_batch_len
            total_acc += acc * cur_batch_len

        return total_loss / len(x_), total_acc / len(x_)

    # 训练与验证
    print('Training and evaluating...')
    start_time = time.time()
    print_per_batch = config.print_per_batch
    best_acc_val = 0.0
    last_improved = 0
    require_improvement = 100

    # 生成批次数据
    print('Generating training batch...')
    batch_train = batch_iter(list(zip(x_train, y_train)),
        config.batch_size, config.num_epochs)

    for i, batch in enumerate(batch_train):
        feed_dict, _ = feed_data(batch)
        feed_dict[model.keep_prob] = config.dropout_keep_prob

        if i % 5 == 0:  # 每5次将训练结果写入tensorboard scalar
            s = session.run(merged_summary, feed_dict=feed_dict)
            writer.add_summary(s, i)

        if i % print_per_batch == print_per_batch - 1:  # 每200次输出在训练集和验证集上的性能
            loss_train, acc_train = session.run([model.loss, model.acc], feed_dict=feed_dict)
            loss_val, acc_val = evaluate(x_val, y_val)

            if acc_val > best_acc_val:
                best_acc_val = acc_val
                last_improved = i
                saver.save(sess=session, save_path=save_path)
                improved_str = '*'
            else:
                improved_str = ''

            time_dif = get_time_dif(start_time)
            msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%},'\
                + ' Val Loss: {3:>6.2}, Val Acc: {4:>7.2%}, Time: {5} {6}'
            print(msg.format(i + 1, loss_train, acc_train, loss_val, acc_val, time_dif, improved_str))

            if i - last_improved > require_improvement:
                print("长时间未提升, 停止优化。")
                break  # 跳出循环

        session.run(model.optim, feed_dict=feed_dict)  # 运行优化

    # 最后在测试集上进行评估
    print('Evaluating on test set...')
    loss_test, acc_test = evaluate(x_test, y_test)
    msg = 'Test Loss: {0:>6.2}, Test Acc: {1:>7.2%}'
    print(msg.format(loss_test, acc_test))

    session.close()

def test_model(cnn=True):
    print('Loading data...')
    start_time = time.time()
    x_test, y_test, words = process_test_file()
    batch_test = batch_iter(list(zip(x_test, y_test)), 128, 1)

    model, save_dir, _ = construct_model(len(words), cnn=cnn, training=False)

    session = tf.Session()
    session.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    save_path = os.path.join(save_dir, 'best_validation')
    saver.restore(sess=session, save_path=save_path)

    total_loss = 0.0
    total_acc = 0.0
    for batch in batch_test:
        x_batch, y_batch = zip(*batch)
        feed_dict = {
            model.input_x: x_batch,
            model.input_y: y_batch,
            model.keep_prob: 1.0
        }
        loss, acc = session.run([model.loss, model.acc], feed_dict=feed_dict)
        total_loss += loss * len(x_batch)
        total_acc += acc * len(x_batch)

    loss_test = total_loss / len(x_test)
    acc_test = total_acc / len(x_test)

    msg = 'Test Loss: {0:>6.2}, Test Acc: {1:>7.2%}'
    print(msg.format(loss_test, acc_test))

    time_dif = get_time_dif(start_time)
    print('Time usage:', time_dif)

def model_score(cnn=True):
    print('Loading data...')
    start_time = time.time()
    x_test, y_test, words = process_test_file()
    batch_test = batch_iter(list(zip(x_test, y_test)), 128, 1)
    y_pred_cls = np.zeros(shape=len(x_test), dtype=np.int32)
    y_test_cls = np.zeros(shape=len(x_test), dtype=np.int32)

    model, save_dir, _ = construct_model(len(words), cnn=cnn, training=False)

    session = tf.Session()
    session.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    save_path = os.path.join(save_dir, 'best_validation')
    saver.restore(sess=session, save_path=save_path)

    for i, batch in enumerate(batch_test):
        x_batch, y_batch = zip(*batch)
        batch_len = len(x_batch)
        feed_dict = {
            model.input_x: x_batch,
            model.keep_prob: 1.0
        }
        y_pred_batch = session.run(model.pred_y, feed_dict=feed_dict)
        y_pred_cls[i*batch_len:(i+1)*batch_len] = np.argmax(y_pred_batch, 1)
        y_test_cls[i*batch_len:(i+1)*batch_len] = np.argmax(y_batch, 1)

    print("Precision, Recall and F1-Score")
    categories, _ = read_category()
    print(metrics.classification_report(y_test_cls, y_pred_cls, target_names=categories))

    print("Confusion Matrix")
    cm = metrics.confusion_matrix(y_test_cls, y_pred_cls)
    print(cm)

    time_dif = get_time_dif(start_time)
    print('Time usage:', time_dif)

if __name__ == '__main__':
    # run_epoch(cnn=True)
    # test_model()
    model_score()
