from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle


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


def train(train_vectors, valid=False, max_depth=30):
    print('开始训练，当前最大深度', max_depth)
    start_time = time.time()
    if valid:
        X = train_vectors.iloc[:, :train_vectors.shape[1] - 1]
        y = train_vectors.iloc[:, train_vectors.shape[1] - 1]
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        clf = DecisionTreeClassifier(max_depth=max_depth)
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        acc = accuracy_score(y_test, y_pred)
        print('正确率为', acc, '%')
    else:
        x_train = train_vectors.iloc[:, :train_vectors.shape[1] - 1]
        y_train = train_vectors.iloc[:, train_vectors.shape[1] - 1]

        clf = DecisionTreeClassifier(random_state=42)
        # 训练模型
        clf.fit(x_train, y_train)
    end_time = time.time() - start_time
    print('训练耗时:', end_time, 's')
    return clf, acc


def plot(clf):
    plt.figure(figsize=(20, 10))

    class_name = ['Room 1', 'Room 2', 'Room 3', 'Room 4']
    plot_tree(clf, filled=True, rounded=True, class_names=class_name)
    plt.show()


def main():
    train_data = preprocess('train_data.txt', label_path='train_labels.txt')
    grid_search(train_data)
    # train_data = preprocess('test_data.txt')
    # plot(clf)


def grid_search(train_data):
    list = []
    for max_depth in range(20, 30, 1):
        l1 = []
        for min_samples_split in range(25, 35, 1):
            l2 = []
            for min_samples_leaf in range(5, 15, 1):
                print(f'当前参数：max_depth={max_depth}, min_samples_split={min_samples_split}, min_sample_leaf={min_samples_leaf}')
                _, acc = train(train_data, True, max_depth)
                l2.append(acc)
            l1.append(l2)
        list.append(l1)
    with open('my_list1 .pkl', 'wb') as f:
        pickle.dump(list, f)
    list = np.array(list)
    print(list)
    # 3, 5, 1

if __name__ == '__main__':
    main()
