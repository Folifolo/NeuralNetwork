import numpy as np
from utils import *

class NN:
    def __init__(self, input_size = 784, hidden_size=30, output_size=10):
        self._input_size = input_size
        self._hidden_size = hidden_size
        self._output_size = output_size

        self._w1 = np.array([])
        self._w2 = np.array([])

        self._b1 = np.array([])
        self._b2 = np.array([])

    def _forward(self, input):
        output1 = self._w1.dot(input.T)+self._b1
        output1 = relu(output1)

        output2 = self._w2.dot(output1) + self._b2
        out2_max = np.max(output2, axis=0)
        exponents = np.exp(output2-out2_max)
        output2 = exponents / exponents.sum(axis = 0)

        return output1, output2

    def _calculate_dE(self, input, label, output1, output2):
        Y_U = label.T - output2
        dE2 = -Y_U.dot(output1.T)
        db2 = -Y_U.dot(np.ones((Y_U.shape[1], 1)))

        wY_U = (self._w2.T).dot(Y_U) * reluD(output1)
        dE1 = -wY_U.dot(input)
        db1 = -wY_U.dot(np.ones((wY_U.shape[1],1)))
        
        return (dE1, db1), (dE2, db2)

    def _backprop(self, learning_rate, size, dEb1, dEb2):
        dE1, db1 = dEb1
        dE2, db2 = dEb2

        self._w2 = self._w2 - learning_rate*dE2/size
        self._b2 = self._b2 - learning_rate*db2/size

        self._w1 = self._w1 - learning_rate*dE1/size
        self._b1 = self._b1 - learning_rate*db1/size

    def init_weights(self):
        self._w1 =np.random.randn(self._hidden_size, self._input_size)/10
        self._w2 =np.random.randn(self._output_size, self._hidden_size)/10

        self._b1 = np.zeros((self._hidden_size,1))
        self._b2 = np.zeros((self._output_size,1))

    def fit(self, input, label, validate_data = None, batch_size = 100, learning_rate = 0.1, epochs = 100):

        for epoch in range(epochs):
            for batch in range(input.shape[0]//batch_size):
                current_X = input[batch*batch_size:(batch+1)*batch_size]
                current_Y = label[batch*batch_size:(batch+1)*batch_size]
                output1, output2 = self._forward(current_X)
                (dE1, db1), (dE2, db2) = self._calculate_dE(current_X, current_Y, output1, output2)

                self._backprop(learning_rate, batch_size, (dE1, db1), (dE2, db2))


            prediction = self.predict(input)
            print("epoch: ", epoch+1)
            print("train accuracy: ", calculate_acc(prediction, label).round(4),
                  "train error: ", calcilate_E(prediction, label).round(4))

            if validate_data != None:

                prediction_val= self.predict(validate_data[0])
                print("validate accuracy: ", calculate_acc(prediction_val, validate_data[1]).round(4),
                      "validate error: ", calcilate_E(prediction_val, validate_data[1]).round(4))
            print("")

    def predict(self, input):
        _, output2 = self._forward(input)
        return output2



