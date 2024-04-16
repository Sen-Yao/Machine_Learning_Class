# 根据用户采集的 WiFi 信息采用决策树预测用户所在房间

电信 2101 林子垚 U202113904

---

## 题目

### 数据集讲解

- 数据集：训练集存于 `TrainDT.csv` 中；测试集存于 `TestDT.csv` 中。 
- BSSIDLabel： BSSID 标识符，每个 AP（接入点，如路由器）拥有 1 个或多个不同的 BSSID，但 1 个 `BSSID` 只属于`1` 个 AP
- RSSLabel：该 BSSID 的信号强度，单位 dbm； 
- RoomLabel: 该 BSSID 被采集时所属的房间号，为类标签，测试集中也含该标签，主要用于计算预测准确度
- SSIDLabel: 该 BSSID 的名称，不唯一； 
- finLabel：finLabel 标号相同，表示这部分 BSSID 在同一时刻被采集到；我们将在同一时刻采集的所有 BSSID 及其相应 RSS 构成的矢量称为一个**指纹** $f_1=[BSSID_1:RSS_1, BSSID_2:RSS_2, \cdots, RoomLabel]$ ；由于 BSSID 的 RSS 在不同位置大小不同，因此指纹可以唯一的标识一个位置。

### 注意

#### 连续值处理

一方面，可以将每个特征划分为两个属性，未接收到 RSS 用 0 表示，接收到 RSS 用 1 表示，则一个样本可表示为

$$f_i=[BSSID_1:1, BSSID_2:0]$$

另一方面可采用二分法对连续属性进行处理，计算每个划分点的信息增益；

#### 特征构造

不同样本中 BSSID 集合不尽相同，因此可以采用所有样本 BSSID 集合的并集作为特征，如指纹 $f_i$ 的 BSSID 集合为 $B_i=\{BSSID_j|BSSID_j\in f_i\}$，则特征可表示为 $B_u=\cup_{i=1}^N B_i$ 。

#### 缺失值处理

本身缺失值也可以作为特征属性；若采用功能二分法则可以填补特殊值 -100 等。


#### 举例说明
$$f_1=[BSSID_1:1,BSSID_2:0,BSSID_3:1,BSSID_4:1,0]$$

$$f_2=[BSSID_1:1,BSSID_2:1,BSSID_3:1,BSSID_4:0,1]$$
	

则 $f_1$ 本身只接收到 $BSSID_1、BSSID_3$ 和 $BSSID_4$ 共 3 个 BSSID ；$f_2$ 本身只接收到 $BSSID_1、BSSID_2$ 和 $BSSID_3$ 共 3 个 BSSID；特征为所有样本 BSSID 的并集 $B_u=\{BSSID_1，BSSID_2，BSSID_3，BSSID_4\}$； 接收到的 BSSID 其值用 1 表示，缺失值用 0 填充；最后一列表示样本类标签，$f_1$ 属于房间 0，$f_2$ 属于房间 1。（上述只是提供一种思路，采用其它方法构造均可以）


所需提交材料：采用训练集对决策树进行训练，使用测试集进行测试，计算精度：预测正确样本数/样本总数。编写实验报告，报告中需要说明数据处理、编程思路等，需要在报告中写明自己模型在测试集中所达到的精度，可以将测试结果部分截图展示在报告中。

## 实验报告

本次实验，我使用了 sklearn，pandas 和 matplotlib 库来实现作业要求。

### 数据处理

由于给的数据本身并不是我们习惯的 特征-标签 格式，因此需要将其进行变换。换而言之，希望得到每个“时刻”下，各种 SSID 的强度以及此时刻用户所处的位置。用这种格式的数据进行决策树训练才是我们需要的。

数据预处理函数如下图

```python
import pandas as pd

def preprocess(train_path, test_path):
    
    print('数据读取中')
    train_df = pd.read_csv(train_path, encoding='GBK')
    test_df = pd.read_csv(test_path, encoding='GBK')
    merged_df = pd.concat([train_df, test_df], axis=0)

    bssid_values = merged_df['BSSIDLabel'].unique().tolist()

    train_vectors = re_vector(bssid_values, train_df)
    print(train_vectors)
    test_vectors = re_vector(bssid_values, test_df)
    print(test_vectors)
    return train_vectors, test_vectors
```

首先，以 GBK 格式，调用 pandas 库中的 read_csv 函数来读取数据。为了将其转换成 特征-标签 格式，首先需要找出在训练集和测试集中出现过的所有 SSID。为此，首先将两个读取到的 DataFrame 合并，然后利用 `.unique()` 找出所有的 SSID，存储到 list `bssid_values`

接下来使用 `re_vector` 函数来实现将数据转换成 特征-标签 格式。具体代码如下

```python
import pandas as pd

def re_vector(bssid_values, df):
    # 初始化向量
    init_vector = [-100] * len(bssid_values) + [0]
    time_vectors = []
    vec = init_vector.copy()
    prev_time = df.iloc[0]['finLabel']

    # 遍历 DataFrame
    for index, row in df.iterrows():
        if row['finLabel'] != prev_time:
            # 时间戳发生变化，将当前向量添加到时间向量列表中
            time_vectors.append(vec)
            # 更新时间戳
            prev_time = row['finLabel']
            # 重置向量
            vec = init_vector.copy()

        # 根据 BSSIDLabel 在向量中更新 RSSLabel
        vec[bssid_values.index(row['BSSIDLabel'])] = row['RSSLabel']
        # 更新 RoomLabel
        vec[-1] = row['RoomLabel']

    # 添加最后一个向量
    time_vectors.append(vec)

    # 创建 DataFrame
    column_names = bssid_values + ['RoomLabel']
    df_time_vectors = pd.DataFrame(time_vectors, columns=column_names)

    return df_time_vectors
```

首先初始化一个默认的向量，长度为 SSID 的种类数加一。此向量的最后一位代表标签，初始化为零；然后对前面的特征进行初始化为 -100，表示默认没有搜索到此 SSID 发射的信号。

然后进入循环，由于原始数据已经按照时间顺序排列整齐，因此只需要遍历所有行，将指定行内的数据归在同一个时刻下，然后对于这一时刻的所有行的数据，按照对应的 SSID 值填写其 RSS 即可。

数据预处理完后，我们得到了 328 个时刻的训练数据，以及 109 个时刻的测试数据，将其返回给主函数即可

### 训练

训练部分，我调用了 sklearn 来实现决策树的训练。具体代码如下

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

def train(train_time_vectors, test_time_vectors):
    x_train = train_time_vectors.iloc[:, :train_time_vectors.shape[1] - 1]
    y_train = train_time_vectors.iloc[:, train_time_vectors.shape[1] - 1]
    x_test = test_time_vectors.iloc[:, :test_time_vectors.shape[1] - 1]
    y_test = test_time_vectors.iloc[:, test_time_vectors.shape[1] - 1]
    clf = DecisionTreeClassifier(random_state=42)
    # 训练模型
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    # 计算准确率
    print('正确率为', accuracy_score(y_test, y_pred) * 100, '%')
    return clf
```

首先，接受数据预处理部分传来的数据，并且分块，然后将决策树类 `DecisionTreeClassifier` 实例化为 `clf`，调用 `fit` 进行训练，并且用 `clf.predict()` 来进行预测和评估。最后输出正确率

### 绘图

为了增加代码的直观性，我调用了 matplotlib 来画出决策树，更能体现决策树的分类过程。绘图函数如下

```python
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

def plot(clf, feature_names):
    plt.figure(figsize=(20, 10))

    class_name = ['Room 1', 'Room 2', 'Room 3', 'Room 4']
    plot_tree(clf, filled=True, rounded=True, class_names=class_name, feature_names=feature_names)
    plt.show()
```

在绘图时需要指定以下各个房间的值，然后调用 `plot_tree` 即可画出此决策树。

### 训练结果和性能

控制台输出

```
正确率为 100.0 %
训练耗时: 1.0633790493011475
```

决策树的图参见 Figure.png