import numpy as np
import pandas as pd
import os
import torch



def load_glove_model(glove_file):
    model = {}
    with open(glove_file, 'r', encoding='utf-8') as f:
        for line in f:
            split_line = line.split()
            word = " ".join(split_line[:-300])
            embedding = np.array([float(val) for val in split_line[-300:]])
            model[word] = embedding
    return model


def get_text_vector(text, glove_model, max_len):
    words = text.split()
    vectors = [glove_model[word] for word in words if word in glove_model]
    # Padding or truncation
    if len(vectors) < max_len:
        vectors += [np.zeros(300)] * (max_len - len(vectors))  # Zero-padding
    else:
        vectors = vectors[:max_len]  # Truncation
    return np.array(vectors)


def save_data(filename, data):
    np.save(filename, data)


def load_data(filename):
    return np.load(filename)


def preprocess():
    # 读取数据
    train_data = pd.read_csv('train.tsv', delimiter='\t', header=None)
    test_data = pd.read_csv('test.tsv', delimiter='\t', header=None)

    train_arg1, train_arg2, train_label = train_data.iloc[:, 2], train_data.iloc[:, 3], train_data.iloc[:, 1]
    test_arg1, test_arg2 = test_data.iloc[:, 2], test_data.iloc[:, 3]

    # 对类别标签编码
    class_dict = {'Comparison': 0, 'Contingency': 1, 'Expansion': 2, 'Temporal': 3}

    # 检查数据是否已被保存
    train_arg1_file = 'input/train_arg1_vector.npy'
    train_arg2_file = 'input/train_arg2_vector.npy'
    test_arg1_file = 'input/test_arg1_vector.npy'
    test_arg2_file = 'input/test_arg2_vector.npy'
    train_label_file = 'input/train_label.npy'

    if not (os.path.exists(train_arg1_file) and os.path.exists(train_arg2_file) and
            os.path.exists(test_arg1_file) and os.path.exists(test_arg2_file) and
            os.path.exists(train_label_file)):
        # 如果文件不存在，进行GloVe向量化并保存结果
        print('Pre-load files are not existed. Loading GloVe model, this may takes a few minutes...')
        # 加载GloVe模型
        glove_model = load_glove_model('glove.840B.300d.txt')
        print('GloVe loaded')
        max_len = 20
        train_arg1_vector = np.array([get_text_vector(text, glove_model, max_len) for text in train_arg1])
        print('train_arg1_vector loaded')
        train_arg2_vector = np.array([get_text_vector(text, glove_model, max_len) for text in train_arg2])
        print('train_arg2_vector loaded')
        test_arg1_vector = np.array([get_text_vector(text, glove_model, max_len) for text in test_arg1])
        print('test_arg1_vector loaded')
        test_arg2_vector = np.array([get_text_vector(text, glove_model, max_len) for text in test_arg2])
        print('test_arg2_vector loaded')
        train_label = np.array([class_dict[label] for label in train_label])
        print('train_label loaded')

        save_data(train_arg1_file, train_arg1_vector)
        save_data(train_arg2_file, train_arg2_vector)
        save_data(test_arg1_file, test_arg1_vector)
        save_data(test_arg2_file, test_arg2_vector)
        save_data(train_label_file, train_label)
    else:
        # 如果文件存在，直接加载数据
        train_arg1_vector = load_data(train_arg1_file)
        train_arg2_vector = load_data(train_arg2_file)
        test_arg1_vector = load_data(test_arg1_file)
        test_arg2_vector = load_data(test_arg2_file)
        train_label = load_data(train_label_file)

    print('To tensor')
    # 转换为Tensor
    train_arg1_tensor = torch.FloatTensor(train_arg1_vector)
    train_arg2_tensor = torch.FloatTensor(train_arg2_vector)
    test_arg1_tensor = torch.FloatTensor(test_arg1_vector)
    test_arg2_tensor = torch.FloatTensor(test_arg2_vector)

    train_label_tensor = torch.LongTensor(train_label)

    train_tensor = torch.cat((train_arg1_tensor, train_arg2_tensor), dim=1)
    test_tensor = torch.cat((test_arg1_tensor, test_arg2_tensor), dim=1)
    return train_label_tensor, train_tensor, test_tensor