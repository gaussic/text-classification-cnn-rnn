#!/usr/bin/python
# -*- coding: utf-8 -*-

from cnn_model import *
from data.cnews_loader import *

import time
from datetime import timedelta

base_dir = 'data/cnews'
vocab_dir = os.path.join(base_dir, 'cnews.vocab.txt')

save_dir = 'checkpoints/textcnn'
save_path = os.path.join(save_dir, 'best_validation')   # 最佳验证结果保存路径

def predict():
    config = TCNNConfig()
    categories, cat_to_id = read_category()
    words, word_to_id = read_vocab(vocab_dir)
    config.vocab_size = len(words)
    model = TextCNN(config)

    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess=session, save_path=save_path)  # 读取保存的模型

    content = "三星ST550以全新的拍摄方式超越了以往任何一款数码相机"
    data = [word_to_id[x] for x in content if x in word_to_id]
    data = [0]*(config.seq_length - len(data)) + data  # 长度固定为600
    data = np.array(data).reshape(1, -1)    # batch_size为1
    feed_dict = {
        model.input_x: data,
        model.keep_prob: 1.0
    }

    y_pred_cls = session.run(model.y_pred_cls, feed_dict=feed_dict)
    print(categories[y_pred_cls[0]])


if __name__ == '__main__':
    predict()
