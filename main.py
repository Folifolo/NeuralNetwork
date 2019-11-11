from keras.datasets import mnist
from keras.utils import np_utils
from Network import NN
import argparse
from datetime import datetime


def createParser ():
    parser = argparse.ArgumentParser()
    parser.add_argument ('--hidden', type= int, default= 30)
    parser.add_argument('--epochs', type= int, default= 20)
    parser.add_argument('--lr', type= float, default= 0.1)
    parser.add_argument('--batch', type= int, default= 100)

    return parser


if __name__ == '__main__':
    parser = createParser()
    namespace = parser.parse_args()
    args = parser.parse_args()

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)

    model = NN(784, args.hidden, 10)
    model.init_weights()

    start = datetime.now()
    model.fit(x_train, y_train, (x_test, y_test), epochs=args.epochs, learning_rate=args.lr, batch_size=args.batch)
    print("Time: ", (datetime.now()-start).total_seconds(), " seconds")



