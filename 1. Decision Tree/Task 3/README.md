# IMDB 数据集电影评测分类（二分类问题）

## 数据集讲解

该数据集是 IMDB 电影数据集的一个子集，已经划分好了测试集和训练集，训练集包括 25000 条电影评论，测试集也有 25000 条，该数据集已经经过预处理，将每条评论的具体单词序列转化为词库里的整数序列，其中每个整数代表该单词在词库里的位置。例如，整数 104 代表该单词是词库的第 104 个单词。为实验简单，词库仅仅保留了 10000 个最常出现的单词，低频词汇被舍弃。每条评论都具有一个标签，0 表示为负面评论，1 表示为正面评论。

训练数据在 `train_data.txt` 文件下，每一行为一条评论，训练集标签在 `train_labels.txt` 文件下，每一行为一条评论的标签；测试数据在 `test_data.txt` 文件下，测试数据标签未给出。
## 思路

这里提供一个最简单的思路，还有其它思路同学们可以自行思考。

首先，需要将每条评论转换为特征向量，这里可以采用 one-hot 编码，举个例子：这里采用的词库大小为 10000，因此转换的 one-hot 编码也是 10000 维的，如某条评论为 `[3, 5]`，则转换得到的 one-hot 编码的 10000 维向量，只有索引为 3 和 5 的元素为1，其余全部为 0。

将每条评论都转换为 one-hot 编码后，再采用决策树算法进行分类。

## 具体要求

将测试数据预测结果，与训练数据标签存储方式相同，存储为 txt 文件，每一行为一条评论的标签。将测试集预测结果的 txt 文件发送到邮箱 zhangzizhuo@hust.edu.cn。实验报告中需要写明具体实验流程，思路等。


---

本次实验，我使用了 sklearn，pandas, numpy 库来实现作业要求。

### 数据预处理

我使用 pandas 作为数据预处理的工具。代码如下

```python
def preprocess(data_path, **kwargs):
    print('正在读取数据')
    vectors = one_hot(data_path)
    if 'label_path' in kwargs:
        labels = []
        with open(kwargs['label_path'], 'r') as file:
            for line in file:
                labels.append(int(line))
        vectors = pd.DataFrame(vectors)
        vectors['Label'] = labels
    return vectors
```

其中，`one_hot` 函数为独热编码的过程

```python
def one_hot(path, vector_size=10000):
    vectors = []
    with open(path, 'r') as file:
        for line in file:
            # 将每行按空格切分并转换为整数
            numbers = list(map(int, line.split()))
            # 创建一个长度为 10000 的零向量
            vector = np.zeros(10000, dtype=int)
            # 对每个数字进行编码
            for number in numbers:
                if 0 <= number < 10000:  # 确保索引在合理范围内
                    vector[number] += 1
            vectors.append(vector)
    return np.array(vectors)
```

此函数首先为每句话创建一个长度为 10000 的零向量，对应了词库中的 10000 个词。如果这句话中出现了某词汇，则在对应的下标加一。由此建立了句子到词向量的映射关系。将此映射关系以 Numpy 数组的形式返回。

然后数据预处理函数再读取 `train_label`，将其中的标签补充到刚刚得到的词向量上，并且命名为 `label`。由此完成了数据预处理部分。


### 训练

在训练决策树的部分，我调用了 sklearn 库进行决策树的训练。具体代码如下

```python
def train(train_vectors, valid=False, max_depth=30):
    print('开始训练，当前最大深度', max_depth)
    start_time = time.time()
    acc = None  # 为了在函数最后能返回acc，确保acc被定义

    X = train_vectors.iloc[:, :train_vectors.shape[1] - 1]
    y = train_vectors.iloc[:, train_vectors.shape[1] - 1]

    if valid:
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        best_ccp_alpha = find_best_ccp_alpha(x_train, y_train)  # 寻找最佳ccp_alpha值

        clf = DecisionTreeClassifier(max_depth=max_depth, ccp_alpha=best_ccp_alpha, random_state=42)
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        acc = accuracy_score(y_test, y_pred)
        print('正确率为', acc, '%')
    else:
        clf = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
        clf.fit(X, y)

    end_time = time.time() - start_time
    print('训练耗时:', end_time, 's')
    return clf, acc
```

这里设置了参数 `vaild` 来判断是否划分验证集，用于监控决策树性能。首先对数据预处理中得到的 DataFrame 进行分割，选出特征和标签。对于需要划分验证集的情况，默认的训练集和验证集的划分比例为 7:3。


然后创造一个决策树 `DecisionTreeClassifier` 实例 `clf`，并且进行训练，然后用测试集进行性能衡量。

### 主函数

综上，本次实验的代码的主函数为

```python
def main():
    train_data = preprocess('train_data.txt', label_path='train_labels.txt')
    test_data = preprocess('test_data.txt')
    clf, _ = train(train_data, True)
    test_y_pred = clf.predict(test_data)
    np.savetxt('test_predictions.txt', test_y_pred, fmt='%d')
```

即先分别读取训练数据，训练标签和测试集数据，然后进行相应的数据预处理函数。调用 `train` 函数进行训练，并输出正确率，最后应用训练得到的决策树，对给定的测试集数据进行预测，最后将结果保存在 `test_predictions.txt`


### 性能

经过测试，决策树在验证集上的正确率约为 72 %