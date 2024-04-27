from preprocess import preprocess
from train import train


def main():
    train_label_tensor, train_tensor, test_tensor = preprocess()
    train(train_label_tensor, train_tensor, test_tensor)


if __name__ == "__main__":
    main()