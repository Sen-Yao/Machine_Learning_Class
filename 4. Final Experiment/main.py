import torch
import argparse
import os

from preprocess import preprocess
from train import train


def main():
    parser = argparse.ArgumentParser(description="A Neural Network for Implicit Discourse Relation Classification ")

    parser.add_argument('--data_path', '-p', type=str, default='data', help='the number of epoch')
    parser.add_argument('--glove_path', '-gp', type=str, default='model/glove.840B.300d.txt', help='the number of epoch')
    parser.add_argument('--max_length', '-ml', type=int, default=20, help='the number of epoch')
    parser.add_argument('--num_epoch', '-e', type=int, default=4, help='the number of epoch')
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.001, help='a optional string')
    parser.add_argument('--hidden_dim', '-hd', type=int, default=64, help='a optional string')
    parser.add_argument('--num_layers', '-l', type=int, default=2, help='a optional string')
    parser.add_argument('--weight_decay', '-wd', type=float, default=0.0001, help='a optional string')

    args = parser.parse_args()

    train_label_tensor, train_tensor, test_tensor = preprocess(args.data_path, args.glove_path, args.max_length)
    model = train(train_label_tensor, train_tensor, test_tensor,
                  args.num_epoch, args.learning_rate, args.hidden_dim, args.num_layers, args.weight_decay)

    try:
        os.mkdir('output')
    except FileExistsError:
        pass
    torch.save(model.state_dict(), 'model/model_state_dict.pth')


if __name__ == "__main__":
    main()
