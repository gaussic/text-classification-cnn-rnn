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
        for line in f:
            try:
                label, content = line.strip().split('\t')
                contents.append(list(content))
                labels.append(label)
            except:
                pass
    return contents, labels

def build_vocab(train_dir, vocab_dir, vocab_size=5000):
    """根据训练集构建词汇表，存储"""
    data_train, _ = _read_file(train_dir)

    all_data = []
    for content in data_train:
        all_data.extend(content)

    counter = Counter(all_data)
    count_pairs = counter.most_common(vocab_size - 1)
    words, _ = list(zip(*count_pairs))
    # 添加一个 <PAD> 来将所有文本pad为同一长度
    words = ['<PAD>'] + list(words)

    open(vocab_dir, 'w', encoding='utf-8').write('\n'.join(words))

def _read_vocab(filename):
    """读取词汇表"""
    words = list(map(lambda line: line.strip(),
        open(filename, 'r', encoding='utf-8').readlines()))
    word_to_id = dict(zip(words, range(len(words))))

    return words, word_to_id

def read_vocab():
    """读取词汇表"""
    vocab_file = open('data/cnews/vocab_cnews.txt', 'r', encoding='utf-8').readlines()
    words = list(map(lambda line: line.strip(),vocab_file))
    word_to_id = dict(zip(words, range(len(words))))

    return words, word_to_id

def read_category():
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
    _, cat_to_id = read_category()
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

def process_file(filename, word_to_id, cat_to_id, max_length=600):
    """将文件转换为id表示"""
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

def preocess_file2(data_path='data/cnews/', seq_length=600):
    """一次性返回所有数据"""
    words, word_to_id = _read_vocab(os.path.join(data_path, 'vocab_cnews.txt'))
    x_train, y_train = _file_to_ids(os.path.join(data_path,
        'cnews.train.txt'), word_to_id, seq_length)
    x_test, y_test = _file_to_ids(os.path.join(data_path,
        'cnews.test.txt'), word_to_id, seq_length)
    x_val, y_val = _file_to_ids(os.path.join(data_path,
        'cnews.val.txt'), word_to_id, seq_length)

    return x_train, y_train, x_test, y_test, x_val, y_val, words

def process_test_file(data_path='data/cnews/', seq_length=600):
    """获取测试数据"""
    words, word_to_id = _read_vocab(os.path.join(data_path, 'vocab_cnews.txt'))
    test_path = os.path.join(data_path, 'cnews.test.txt')
    x_test, y_test = _file_to_ids(test_path, word_to_id, seq_length)
    return x_test, y_test, words

def batch_iter(x, y, batch_size=64):
    """生成批次数据"""
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1

    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = x[indices]
    y_shuffle = y[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]


if __name__ == '__main__':
    if not os.path.exists('data/cnews/vocab_cnews.txt'):
        _build_vocab('data/cnews/cnews.train.txt')
#     x_train, y_train, x_test, y_test, x_val, y_val = preocess_file()
#     print(x_train.shape, y_train.shape)
#     print(x_test.shape, y_test.shape)
#     print(x_val.shape, y_val.shape)
