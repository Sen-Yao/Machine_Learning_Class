import pandas as pd
import os
import torch
import numpy as np
import time


def load_glove_model(glove_file):
    """
    Load glove model more efficiently using a single pass for line reading and processing.
    :param glove_file: the path to glove file
    :return: A dictionary mapping words to their embeddings.
    """

    vocab = []
    embeddings = []
    with open(glove_file, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 301:
                vocab.append(parts[0])
                embeddings.append(np.array(parts[1:], dtype=float))

    embeddings = np.array(embeddings)

    model = {word: embeddings[i] for i, word in enumerate(vocab)}
    return model


def get_text_vector(text, glove_model, max_len):
    """
    Use Glove model to transfer text to a series of word vector.
    :param text: input English sentence
    :param glove_model: The glove dictionary
    :param max_len: The maximum length for each sentence. The sentences that shorter than that will be fill with zero
    and sentences longer than that will be truncated.
    :return: 2-D NumPy array of text vector
    """
    words = text.split()
    vectors = []
    for word in words:
        if word in glove_model:
            vectors.append(glove_model[word])
        else:
            # Word that can not be found in GloVe
            print(f"Word not found in GloVe data: {word}")
            # Filled with zero
            vectors.append(np.zeros(300))
    # Padding or truncation
    if len(vectors) < max_len:
        vectors += [np.zeros(300)] * (max_len - len(vectors))  # Zero-padding
    else:
        vectors = vectors[:max_len]  # Truncation
    return np.array(vectors)


def get_vector(glove_path, max_length, train_arg1, train_arg2, train_label, test_arg1, test_arg2):
    class_dict = {'Comparison': 0, 'Contingency': 1, 'Expansion': 2, 'Temporal': 3}
    try:
        os.mkdir('input')
    except FileExistsError:
        pass
    train_arg1_file = 'input/train_arg1_vector.npy'
    train_arg2_file = 'input/train_arg2_vector.npy'
    test_arg1_file = 'input/test_arg1_vector.npy'
    test_arg2_file = 'input/test_arg2_vector.npy'
    train_label_file = 'input/train_label.npy'

    if not (os.path.exists(train_arg1_file) and os.path.exists(train_arg2_file) and
            os.path.exists(test_arg1_file) and os.path.exists(test_arg2_file) and
            os.path.exists(train_label_file)):

        print(f'Pre-load files are not existed. Loading GloVe model, this may takes a few minutes...')
        glove_model = load_glove_model(glove_path)
        print('GloVe loaded')
        train_arg1_vector = np.array([get_text_vector(text, glove_model, max_length) for text in train_arg1])
        print('train_arg1_vector loaded')
        train_arg2_vector = np.array([get_text_vector(text, glove_model, max_length) for text in train_arg2])
        print('train_arg2_vector loaded')
        test_arg1_vector = np.array([get_text_vector(text, glove_model, max_length) for text in test_arg1])
        print('test_arg1_vector loaded')
        test_arg2_vector = np.array([get_text_vector(text, glove_model, max_length) for text in test_arg2])
        print('test_arg2_vector loaded')
        train_label = np.array([class_dict[label] for label in train_label])
        print('train_label loaded')

        np.save(train_arg1_file, train_arg1_vector)
        np.save(train_arg2_file, train_arg2_vector)
        np.save(test_arg1_file, test_arg1_vector)
        np.save(test_arg2_file, test_arg2_vector)
        np.save(train_label_file, train_label)
    else:
        train_arg1_vector = np.load(train_arg1_file)
        train_arg2_vector = np.load(train_arg2_file)
        test_arg1_vector = np.load(test_arg1_file)
        test_arg2_vector = np.load(test_arg2_file)
        train_label = np.load(train_label_file)
    return train_arg1_vector, train_arg2_vector, test_arg1_vector, test_arg2_vector, train_label


def preprocess(path, glove_path, max_length):
    """
    preprocess data for training
    :param path: input data's folder.
    :param glove_path: path to glove model
    :param max_length: The maximum length for each sentence. The sentences that shorter than that will be fill with zero
    and sentences longer than that will be truncated.
    :return: training labels, training vectors and testing vectors
    """
    train_path = os.path.join(path, 'train.tsv')
    test_path = os.path.join(path, 'test.tsv')

    train_data = pd.read_csv(train_path, delimiter='\t', header=None)
    test_data = pd.read_csv(test_path, delimiter='\t', header=None)

    train_arg1, train_arg2, train_label = train_data.iloc[:, 2], train_data.iloc[:, 3], train_data.iloc[:, 1]
    test_arg1, test_arg2 = test_data.iloc[:, 2], test_data.iloc[:, 3]

    train_arg1_vector, train_arg2_vector, test_arg1_vector, test_arg2_vector, train_label = get_vector(glove_path, max_length, train_arg1, train_arg2, train_label, test_arg1, test_arg2)

    train_arg1_tensor = torch.FloatTensor(train_arg1_vector)
    train_arg2_tensor = torch.FloatTensor(train_arg2_vector)
    test_arg1_tensor = torch.FloatTensor(test_arg1_vector)
    test_arg2_tensor = torch.FloatTensor(test_arg2_vector)
    train_label_tensor = torch.LongTensor(train_label)

    train_tensor = torch.cat((train_arg1_tensor, train_arg2_tensor), dim=1)
    test_tensor = torch.cat((test_arg1_tensor, test_arg2_tensor), dim=1)
    return train_label_tensor, train_tensor, test_tensor
