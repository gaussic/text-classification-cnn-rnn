#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import Counter
import tensorflow.contrib.keras as kr
import numpy as np
import os

def _read_file(filename):
    """读取文件数据"""
    contents = []
    labels = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            try:
                label, content = line.strip().split('\t')
                contents.append(list(content))
                labels.append(label)
            except:
                pass
    return contents, labels

def _build_vocab(filename, vocab_size=5000):
    """根据训练集构建词汇表，存储"""
    data, _ = _read_file(filename)

    all_data = []
    for content in data:
        all_data.extend(content)

    counter = Counter(all_data)
    count_pairs = counter.most_common(vocab_size - 1)
    words, _ = list(zip(*count_pairs))
    # 添加一个 <PAD> 来将所有文本pad为同一长度
    words = ['<PAD>'] + list(words)

    open('data/cnews/vocab_cnews.txt',
        'w', encoding='utf-8').write('\n'.join(words))

def _read_vocab(filename):
    """读取词汇表"""
    words = list(map(lambda line: line.strip(),
        open(filename, 'r', encoding='utf-8').readlines()))
    word_to_id = dict(zip(words, range(len(words))))

    return words, word_to_id

def _read_category():
    """读取分类目录，固定"""
    categories = ['体育', '财经', '房产', '家居',
        '教育', '科技', '时尚', '时政', '游戏', '娱乐']
    cat_to_id = dict(zip(categories, range(len(categories))))

    return categories, cat_to_id

def to_words(content, words):
    """将id表示的内容转换为文字"""
    return ''.join(words[x] for x in content)

def _file_to_ids(filename, word_to_id, max_length=600):
    """将文件转换为id表示"""
    _, cat_to_id = _read_category()
    contents, labels = _read_file(filename)

    data_id = []
    label_id = []
    for i in range(len(contents)):
        data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])
        label_id.append(cat_to_id[labels[i]])

    # 使用keras提供的pad_sequences来将文本pad为固定长度
    x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length)
    y_pad = kr.utils.to_categorical(label_id)  # 将标签转换为one-hot表示

    return x_pad, y_pad

def preocess_file(data_path='data/cnews/'):
    """一次性返回所有数据"""
    _, word_to_id = _read_vocab(os.path.join(data_path,
        'vocab_cnews.txt'))
    x_train, y_train = _file_to_ids(os.path.join(data_path,
        'cnews.train.txt'), word_to_id)
    x_test, y_test = _file_to_ids(os.path.join(data_path,
        'cnews.test.txt'), word_to_id)
    x_val, y_val = _file_to_ids(os.path.join(data_path,
        'cnews.val.txt'), word_to_id)

    return x_train, y_train, x_test, y_test, x_val, y_val

def batch_iter(data, batch_size=64, num_epochs=5):
    """生成批次数据"""
    data = np.array(data)
    data_size = len(data)
    num_batchs_per_epoch = int((data_size - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[indices]

        for batch_num in range(num_batchs_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


if __name__ == '__main__':

    if not os.path.exists('data/cnews/vocab_cnews.txt'):
        _build_vocab('data/cnews/cnews.train.txt')
    x_train, y_train, x_test, y_test, x_val, y_val = preocess_file()
    print(x_train.shape, y_train.shape)
    print(x_test.shape, y_test.shape)
    print(x_val.shape, y_val.shape)
