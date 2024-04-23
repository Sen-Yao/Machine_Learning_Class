import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import accuracy_score, f1_score

# 读取数据
train_data = pd.read_csv('train.tsv', delimiter='\t', header=None)
test_data = pd.read_csv('test.tsv', delimiter='\t', header=None)

train_arg1, train_arg2, train_label = train_data.iloc[:, 2], train_data.iloc[:, 3], train_data.iloc[:, 1]
test_arg1, test_arg2 = test_data.iloc[:, 2], test_data.iloc[:, 3]

# 对类别标签编码
class_dict = {'Comparison': 0, 'Contingency': 1, 'Expansion': 2, 'Temporal': 3}
train_label = np.array([class_dict[label] for label in train_label])

# 初始化TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')

# 训练Vectorizer并转换数据
train_arg1_vector = vectorizer.fit_transform(train_arg1)
train_arg2_vector = vectorizer.fit_transform(train_arg2)
test_arg1_vector = vectorizer.transform(test_arg1)
test_arg2_vector = vectorizer.transform(test_arg2)

# 将稀疏矩阵转换为Dense矩阵，因SVM不支持稀疏输入
train_arg1_feature = train_arg1_vector.toarray()
train_arg2_feature = train_arg2_vector.toarray()
test_arg1_feature = test_arg1_vector.toarray()
test_arg2_feature = test_arg2_vector.toarray()

train_feature = np.concatenate((train_arg1_feature, train_arg2_feature), axis=1)
test_feature = np.concatenate((test_arg1_feature, test_arg2_feature), axis=1)

print('Start training')

# SVM分类
clf = svm.SVC(decision_function_shape='ovo')
clf.fit(train_feature, train_label)

print('Start predicting')

# 计算训练集上的Acc和F1
train_pred = clf.predict(train_feature)
train_acc = accuracy_score(train_label, train_pred)
train_f1 = f1_score(train_label, train_pred, average='macro')
print(f'Train Set: Acc={train_acc:.4f}, F1={train_f1:.4f}')

# 计算测试集预测结果并保存
test_pred = clf.predict(test_feature)
with open('test_pred.txt', 'w') as f:
    for label in tqdm(test_pred):
        f.write(str(label) + '\n')
f.close()
