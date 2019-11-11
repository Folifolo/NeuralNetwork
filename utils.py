import numpy as np

def relu(X):
    return X * (X > 0)

def reluD(X):
    return X > 0

#=====================

def calcilate_E(predict, label):
    lgr = np.log(predict.T)
    return -np.sum(label * lgr) / label.shape[0]

def calculate_acc(prediction ,label):
    prediction = np.argmax(prediction, axis= 0)
    label = np.argmax(label, axis= 1)
    return (prediction == label).mean()
