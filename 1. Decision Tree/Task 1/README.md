# 使用决策树预测隐形眼镜类型


电信 2101 林子垚 U202113904

---

## 问题

眼科医生是如何判断患者需要佩戴的镜片类型的？ 隐形眼镜数据集是非常著名的数据集，它包含了很多患者眼部状况的观察条件以及医生推荐的隐形眼镜类型。隐形眼镜类型包括硬材质（hard）、软材质（soft）以及不适合佩戴隐形眼镜（no lenses）。以下为该数据集的部分数据，包括年龄、近视 or 远视类型，是否散光，是否容易流泪，最后 1 列为应佩戴眼镜类型

## 要求

- 准备数据：用 Python 解析文本文件，解析 tab 键分割的数据行
- 分析数据：快速检查数据，确保正确地解析数据内容；训练算法
- 采用决策树分类算法，获得预测隐形眼镜类型的决策树。
- 所需提交材料：任务一需要编程画出决策树，编写实验报告进行简述其原理，编程思路等。

## 实验报告

本次实验，我使用了 sklearn，pandas 和 matplotlib 库来实现作业要求。

### 数据预处理

我使用 pandas 作为数据预处理的工具。代码如下

```python
import pandas as pd

def preprocess(path, map_list):

    df = pd.read_csv(path, sep='\t', header=None)

    for column in range(df.shape[1]):
        df[column] = df[column].map(map_list[column])

    return df
```

首先，调用 pandas 的 read_csv 来读取文件。由于数据集 `lenses.txt` 是用制表符分割的，且没有标题行，因此指定参数 `sep='\t', header=None`。得到 pandas 的 DataFrame 格式数据。

在主函数中，我制定了各个数据的映射方式

```python
mapping_list = [{'young': 0, 'pre': 1, 'presbyopic': 2},
                    {'myope': 0, 'hyper': 1},
                    {'no': 0, 'yes': 1},
                    {'reduced': 0, 'normal': 1},
                    {'no lenses': 0, 'soft': 1, 'hard': 2}]
```

将此参数传入 `preprocess` 函数，然后对数据的每一列进行遍历，通过 `map` 函数即可将自然语言数据转换为 float 类型数据，方便决策树的计算。

### 训练

在训练决策树的部分，我调用了 sklearn 库进行决策树的训练。具体代码如下

```python
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

def train(df): 
    X = df.iloc[:, :df.shape[1] - 1]
    y = df.iloc[:, df.shape[1] - 1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = DecisionTreeClassifier(random_state=42)
    # 训练模型
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    # 计算准确率
    print('正确率为', accuracy_score(y_test, y_pred) * 100, '%')
    return clf
```

首先对数据预处理中得到的 DataFrame 进行分割，选出特征和标签。

然后调用 `train_test_split` 函数进行训练集和测试集的划分。其中指定了参数 `test_size=0.2, random_state=42`。前者用于划分数据集中测试集的比例，后者用于控制随机性，保证可复制性。

然后创造一个决策树 `DecisionTreeClassifier` 实例 `clf`，并且进行训练，然后用测试集进行性能衡量。输出结果为“正确率为 100.0 %”

### 绘图

为了增加代码的直观性，我调用 matplotlib 进行绘图。具体代码如下

```python
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

def plot(clf):
    feature_names = ['age', 'prescription', 'astigmatic', 'tear']
    class_names = ['no lenses', 'soft', 'hard']
    plt.figure(figsize=(20, 10))


    plot_tree(clf, filled=True, rounded=True, class_names=class_names, feature_names=feature_names)
    plt.show()
```

首先为了绘图时能够展示数据的特征和标签，手动输入特征和标签所对应的自然语言。然后由此指定 `class_names=class_names, feature_names=feature_names` 来绘图。绘图结果参见 [Figure.png]

### 性能

经过测试，正确率为 100.0 %