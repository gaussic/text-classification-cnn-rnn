# Text Classification with CNN

使用卷积神经网络进行文本分类

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

## 模型


```
Loading data...
Time usage: 0:00:16
Constructing Model...
Training and evaluating...
Iter:      1, Train Loss:    2.3, Train Acc:  10.94%, Val Loss:    2.3, Val Acc:  10.06%, Time: 0:00:01
Iter:    201, Train Loss:   0.37, Train Acc:  87.50%, Val Loss:   0.58, Val Acc:  81.70%, Time: 0:00:06
Iter:    401, Train Loss:   0.22, Train Acc:  90.62%, Val Loss:   0.34, Val Acc:  91.16%, Time: 0:00:11
Iter:    601, Train Loss:   0.17, Train Acc:  95.31%, Val Loss:   0.28, Val Acc:  92.16%, Time: 0:00:16
Iter:    801, Train Loss:   0.18, Train Acc:  95.31%, Val Loss:   0.25, Val Acc:  93.12%, Time: 0:00:21
Iter:   1001, Train Loss:   0.12, Train Acc:  95.31%, Val Loss:   0.28, Val Acc:  91.52%, Time: 0:00:26
Iter:   1201, Train Loss:  0.085, Train Acc:  96.88%, Val Loss:   0.24, Val Acc:  92.92%, Time: 0:00:31
Iter:   1401, Train Loss:  0.098, Train Acc:  95.31%, Val Loss:   0.22, Val Acc:  93.40%, Time: 0:00:36
Iter:   1601, Train Loss:  0.042, Train Acc:  98.44%, Val Loss:   0.19, Val Acc:  94.70%, Time: 0:00:41
Iter:   1801, Train Loss:  0.035, Train Acc: 100.00%, Val Loss:   0.19, Val Acc:  94.88%, Time: 0:00:46
Iter:   2001, Train Loss:  0.011, Train Acc: 100.00%, Val Loss:    0.2, Val Acc:  94.38%, Time: 0:00:51
Iter:   2201, Train Loss:    0.1, Train Acc:  96.88%, Val Loss:    0.2, Val Acc:  94.60%, Time: 0:00:56
Iter:   2401, Train Loss:  0.015, Train Acc: 100.00%, Val Loss:   0.19, Val Acc:  94.88%, Time: 0:01:01
Iter:   2601, Train Loss:  0.017, Train Acc: 100.00%, Val Loss:   0.21, Val Acc:  94.24%, Time: 0:01:06
Iter:   2801, Train Loss: 0.0014, Train Acc: 100.00%, Val Loss:   0.17, Val Acc:  95.36%, Time: 0:01:11
Iter:   3001, Train Loss:  0.074, Train Acc:  98.44%, Val Loss:   0.23, Val Acc:  94.02%, Time: 0:01:17
Iter:   3201, Train Loss:  0.033, Train Acc:  98.44%, Val Loss:   0.15, Val Acc:  96.26%, Time: 0:01:22
Iter:   3401, Train Loss: 0.0087, Train Acc: 100.00%, Val Loss:    0.2, Val Acc:  94.38%, Time: 0:01:27
Iter:   3601, Train Loss: 0.0088, Train Acc: 100.00%, Val Loss:   0.23, Val Acc:  94.16%, Time: 0:01:32
Iter:   3801, Train Loss:  0.015, Train Acc: 100.00%, Val Loss:   0.26, Val Acc:  93.08%, Time: 0:01:37
Test Loss:   0.17, Test Acc:  95.72%
```
