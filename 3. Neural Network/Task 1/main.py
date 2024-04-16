import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def read(path):
    data = []
    with open(path, 'r') as file:
        for line in file:
            numbers = list(map(float, line.split()))
            data.append(numbers)
    return pd.DataFrame(data)


def train(train_data, test_data):
    x_train = train_data.iloc[:, :train_data.shape[1] - 1]
    y_train = train_data.iloc[:, -1]
    x_test = test_data.iloc[:, :test_data.shape[1] - 1]
    y_test = test_data.iloc[:, -1]
    scaler = StandardScaler()
    X_train = scaler.fit_transform(x_train)
    X_test = scaler.transform(x_test)

    log_reg = LogisticRegression()
    log_reg.fit(X_train, y_train)
    y_pred = log_reg.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:",  100 * accuracy, '%')


def main():
    train_data = read('horseColicTraining.txt')
    test_data = read('horseColicTest.txt')
    train(train_data, test_data)


if __name__ == '__main__':
    main()
