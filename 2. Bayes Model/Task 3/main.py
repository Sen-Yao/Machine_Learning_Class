from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import time


def one_hot(path):
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


def train(train_vectors, valid=False):
    start_time = time.time()
    acc = None  # 为了在函数最后能返回acc，确保acc被定义

    X = train_vectors.iloc[:, :train_vectors.shape[1] - 1]
    y = train_vectors.iloc[:, train_vectors.shape[1] - 1]

    if valid:
        x_train, x_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=42)

        clf = MultinomialNB()
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_valid)
        acc = accuracy_score(y_valid, y_pred)
        print('正确率为', 100 * acc, '%')
    else:
        clf = MultinomialNB()
        clf.fit(X, y)

    end_time = time.time() - start_time
    print('训练耗时:', end_time, 's')
    return clf, acc


def main():
    train_data = preprocess('train/train_data.txt', label_path='train/train_labels.txt')
    test_data = preprocess('test/test_data.txt')
    clf, _ = train(train_data, True)
    test_y_pred = clf.predict(test_data)
    print(test_y_pred)
    np.savetxt('test_predictions.txt', test_y_pred, fmt='%d')


if __name__ == '__main__':
    main()
