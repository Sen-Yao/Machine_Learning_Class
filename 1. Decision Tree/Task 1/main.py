from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt


def preprocess(path, map_list):

    df = pd.read_csv(path, sep='\t', header=None)

    for column in range(df.shape[1]):
        df[column] = df[column].map(map_list[column])

    return df

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

def plot(clf):
    feature_names = ['age', 'prescription', 'astigmatic', 'tear']
    class_names = ['no lenses', 'soft', 'hard']
    plt.figure(figsize=(20, 10))


    plot_tree(clf, filled=True, rounded=True, class_names=class_names, feature_names=feature_names)
    plt.show()


def main():
    mapping_list = [{'young': 0, 'pre': 1, 'presbyopic': 2},
                    {'myope': 0, 'hyper': 1},
                    {'no': 0, 'yes': 1},
                    {'reduced': 0, 'normal': 1},
                    {'no lenses': 0, 'soft': 1, 'hard': 2}]
    df = preprocess('lenses.txt', mapping_list)
    clf = train(df)
    plot(clf)


if __name__ == '__main__':
    main()
