# 任务一：使用朴素贝叶斯过滤垃圾邮件

## 一、问题

现有 50 封电子邮件，存放在数据集中，试基于朴素贝叶斯分类器原理，用 Python 编程实现对垃圾邮件和正常邮件的分类。采用交叉验证方式并且输出分类的错误率及分类错误的文档。

## 二、步骤提示

### 1. 收集数据

这里数据直接提供，一般情况下需要自己手动采集及预分类；

### 2. 数据管理

拿到手的数据并不能直接使用，我们需要对数据进行预处理，变成向量的形式（这里采用**词袋模型**）。可以首先把所有字符转换成小写，去掉大小字符不统一的影响；然后构建一个包含在所有文档中出现的不重复的词的列表；获得词汇表后，根据某个单词在一篇文档中出现的次数获得文档向量（这三步的代码可参考给出的 demo）；最后利用构建的分类器进行训练。

### 3. 训练和测试

导入文件夹 spam 与 ham 下的文本文件，并将它们解析为词列表。接下来构建一个训练集和测试集，两个集合中的邮件都是随机选出的。经过一次迭代就能输出分类的错误率及分类错误的文档。（注：由于测试集的选择是随机的，所以测试算法时每次的输出结果可能有些差别，如果想要更好的估计错误率，最好将上述过程重复多次，然后求平均值。）

## 三、提交要求

任务一需要编写实验报告进行简述其原理，编程思路，数据集的划分方式以及错误率等。
