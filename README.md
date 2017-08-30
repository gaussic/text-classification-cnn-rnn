# Text Classification with CNN

使用卷积神经网络进行文本分类

CNN做句子分类的论文可以参看: [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)

还可以去读dennybritz大牛的博客：[Implementing a CNN for Text Classification in TensorFlow](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/)

本文是基于TensorFlow在中文数据集上的另一种实现，如果你觉得对你有帮助，欢迎star与交流。

## 数据集

使用THUCNews的一个子集进行训练与测试，数据集请自行到[THUCTC：一个高效的中文文本分类工具包](http://thuctc.thunlp.org/)下载，请遵循数据提供方的开源协议。

本次训练使用了其中的10个分类，每个分类6500条数据。

类别如下：

```
体育, 财经, 房产, 家居, 教育, 科技, 时尚, 时政, 游戏, 娱乐
```

数据集划分如下：

- 训练集: 5000*10
- 验证集: 500*10
- 测试集: 1000*10

从原数据集生成子集的过程请参看`helper`下的两个脚本。其中，`copy_data.sh`用于从每个分类拷贝6500个文件，`cnews_group.py`用于将多个文件整合到一个文件中。执行该文件后，得到三个数据文件：

- cnews.train.txt: 训练集(50000条)
- cnews.val.txt: 验证集(5000条)
- cnews.test.txt: 测试集(10000条)

## 预处理

`data/cnews_loader.py`为数据的预处理文件。

- `read_file()`：读取上一部分生成的数据文件，将内容和标签分开返回;
- `_build_vocab()`: 构建词汇表，这里不需要对文档进行分词，单字的效果已经很好，这一函数会将词汇表存储下来，避免每一次重复处理;
- `_read_vocab()`: 读取上一步存储的词汇表，转换为`{词：id}`表示;
- `_read_category()`: 将分类目录固定，转换为`{类别: id}`表示;
- `_file_to_ids()`: 基于上面定义的函数，将数据集从文字转换为id表示;
- `to_words()`: 将一条由id表示的数据重新转换为文字;
- `preocess_file()`: 一次性处理所有的数据并返回;
- `batch_iter()`: 为神经网络的训练准备批次的数据。

经过数据预处理，数据的格式如下：

| Data | Shape | Data | Shape |
| :---------- | :---------- | :---------- | :---------- |
| x_train | [50000, 600] | y_train | [50000, 10] |
| x_val | [5000, 600] | y_val | [5000, 10] |
| x_test | [10000, 600] | y_test | [10000, 10] |

## 配置项

可配置的参数如下所示，在`model.py`的上部。

```
class TCNNConfig(object):
    """配置参数"""

    # 模型参数
    embedding_dim = 64      # 词向量维度
    seq_length = 600        # 序列长度
    num_classes = 10        # 类别数
    num_filters = 256       # 卷积核数目
    kernel_size = 5         # 卷积核尺寸
    vocab_size = 5000       # 词汇表达小

    hidden_dim = 128        # 全链接层神经元

    dropout_keep_prob = 0.8 # dropout保留比例
    learning_rate = 1e-3    # 学习率

    batch_size = 128         # 每批训练大小
    num_epochs = 10          # 总迭代轮次
```

## 模型

原始的模型如下图所示：

![raw](images/raw_cnn_architecture.png)

可看到它使用了多个不同宽度的卷积核然后将它们做了一个max over time pooling转换为一个长的特征向量，再使用softmax进行分类。

实验发现，简单的cnn也能达到较好的效果。

因此在这里使用的是简化版的结构，具体参看`model.py`。

首先在初始化时，需要定义两个`placeholder`作为输入输出占位符。

```
def __init__(self, config):
      self.config = config

      self.input_x = tf.placeholder(tf.int32,
          [None, self.config.seq_length], name='input_x')
      self.input_y = tf.placeholder(tf.float32,
          [None, self.config.num_classes], name='input_y')

      self.cnn()
```

词嵌入将词的id映射为词向量表示，embedding层会在训练时更新。

```
def input_embedding(self):
    """词嵌入"""
    with tf.device('/cpu:0'):
        embedding = tf.get_variable('embedding',
            [self.config.vocab_size, self.config.embedding_dim])
        _inputs = tf.nn.embedding_lookup(embedding, self.input_x)
    return _inputs
```

cnn模型中，首先定义一个一维卷积层，再使用`tf.reduce_max`实现global max pooling。再接两个dense层分别做映射和分类。使用交叉熵损失函数，Adam优化器，并且计算准确率。这里有许多参数可调，大部分可以通过调整TCNNConfig类即可。

```
def cnn(self):
      """cnnc模型"""
      embedding_inputs = self.input_embedding()

      with tf.name_scope("cnn"):
          # cnn 与全局最大池化
          conv = tf.layers.conv1d(embedding_inputs,
              self.config.num_filters,
              self.config.kernel_size, name='conv')

          # global max pooling
          gmp = tf.reduce_max(conv, reduction_indices=[1], name='gmp')

      with tf.name_scope("score"):
          # 全连接层，后面接dropout以及relu激活
          fc = tf.layers.dense(gmp, self.config.hidden_dim, name='fc1')
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
```

## 训练与验证

这一部分详见代码，具体不在此叙述。

在设定迭代轮次为10的时候，测试集达到了96.67%的准确率，可见效果还是很理想的。

```
Loading data...
Time usage: 0:00:16
Constructing Model...
Training and evaluating...
Iter:      1, Train Loss:    2.3, Train Acc:  10.16%, Val Loss:    2.3, Val Acc:  10.54%, Time: 0:00:01
Iter:    201, Train Loss:   0.24, Train Acc:  91.41%, Val Loss:   0.43, Val Acc:  87.94%, Time: 0:00:10
Iter:    401, Train Loss:  0.075, Train Acc:  96.88%, Val Loss:   0.29, Val Acc:  91.54%, Time: 0:00:20
Iter:    601, Train Loss:    0.1, Train Acc:  96.88%, Val Loss:   0.26, Val Acc:  92.32%, Time: 0:00:29
Iter:    801, Train Loss:   0.13, Train Acc:  92.97%, Val Loss:   0.26, Val Acc:  92.82%, Time: 0:00:38
Iter:   1001, Train Loss:  0.098, Train Acc:  95.31%, Val Loss:   0.22, Val Acc:  93.36%, Time: 0:00:47
Iter:   1201, Train Loss:  0.025, Train Acc:  98.44%, Val Loss:   0.21, Val Acc:  93.42%, Time: 0:00:56
Iter:   1401, Train Loss: 0.0046, Train Acc: 100.00%, Val Loss:   0.18, Val Acc:  94.60%, Time: 0:01:05
Iter:   1601, Train Loss:  0.031, Train Acc:  99.22%, Val Loss:   0.19, Val Acc:  94.68%, Time: 0:01:14
Iter:   1801, Train Loss:  0.018, Train Acc:  99.22%, Val Loss:    0.2, Val Acc:  94.30%, Time: 0:01:23
Iter:   2001, Train Loss: 0.0069, Train Acc: 100.00%, Val Loss:   0.21, Val Acc:  94.10%, Time: 0:01:33
Iter:   2201, Train Loss: 0.0035, Train Acc: 100.00%, Val Loss:   0.21, Val Acc:  94.70%, Time: 0:01:42
Iter:   2401, Train Loss: 0.0079, Train Acc: 100.00%, Val Loss:   0.22, Val Acc:  93.86%, Time: 0:01:51
Iter:   2601, Train Loss: 0.0034, Train Acc: 100.00%, Val Loss:   0.22, Val Acc:  94.46%, Time: 0:02:00
Iter:   2801, Train Loss: 0.00073, Train Acc: 100.00%, Val Loss:   0.24, Val Acc:  93.76%, Time: 0:02:09
Iter:   3001, Train Loss: 0.0026, Train Acc: 100.00%, Val Loss:   0.23, Val Acc:  94.22%, Time: 0:02:18
Iter:   3201, Train Loss: 0.0012, Train Acc: 100.00%, Val Loss:   0.22, Val Acc:  94.68%, Time: 0:02:27
Iter:   3401, Train Loss: 0.0025, Train Acc: 100.00%, Val Loss:   0.22, Val Acc:  94.52%, Time: 0:02:36
Iter:   3601, Train Loss: 0.0016, Train Acc: 100.00%, Val Loss:   0.23, Val Acc:  94.92%, Time: 0:02:45
Iter:   3801, Train Loss: 0.0052, Train Acc: 100.00%, Val Loss:   0.31, Val Acc:  93.30%, Time: 0:02:54
Evaluating on test set...
Test Loss:   0.15, Test Acc:  96.67%
```

准确率和误差如图所示：

![images/acc_loss.png](images/acc_loss.png)
