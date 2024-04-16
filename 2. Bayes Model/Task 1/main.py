from preprocess import preprocess
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import pandas as pd


def train(ham_vectors, spam_vectors):
    ham_vectors = pd.DataFrame(ham_vectors)
    zeros_vector = np.zeros(ham_vectors.shape[0])
    spam_vectors = pd.DataFrame(spam_vectors)
    ones_vector = np.ones(spam_vectors.shape[0])

    ham_train_x, ham_test_x, ham_train_y, ham_test_y = train_test_split(ham_vectors, zeros_vector, test_size=0.2, random_state=1)
    spam_train_x, spam_test_x, spam_train_y, spam_test_y = train_test_split(spam_vectors, ones_vector, test_size=0.2, random_state=0)

    train_x = pd.concat([ham_train_x, spam_train_x])
    train_y = np.concatenate([ham_train_y, spam_train_y])

    # Initialize and train the Naive Bayes classifier
    clf = MultinomialNB()
    clf.fit(train_x, train_y)

    # Evaluate the classifier
    ham_accuracy = clf.score(ham_test_x, ham_test_y)
    spam_accuracy = clf.score(spam_test_x, spam_test_y)
    print("Ham Accuracy:", ham_accuracy * 100, '%')
    print("Spam Accuracy:", spam_accuracy * 100, '%')


def main():
    ham_vectors, spam_vectors = preprocess()
    train(ham_vectors, spam_vectors)


if __name__ == '__main__':
    main()