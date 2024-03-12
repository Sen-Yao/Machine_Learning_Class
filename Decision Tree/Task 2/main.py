from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
import time


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


def plot(clf, feature_names):
    plt.figure(figsize=(20, 10))

    class_name = ['Room 1', 'Room 2', 'Room 3', 'Room 4']
    plot_tree(clf, filled=True, rounded=True, class_names=class_name, feature_names=feature_names)
    plt.show()


def main():
    start_time = time.time()
    train_time_vectors, test_time_vectors = preprocess('TrainDT.csv', 'TestDT.csv')
    clf = train(train_time_vectors, test_time_vectors)
    end_time = time.time() - start_time
    print('训练耗时:', end_time, 's')
    feature_names = test_time_vectors.columns.tolist()
    plot(clf, feature_names)



if __name__ == '__main__':
    main()
