from __future__ import print_function
import numpy as np
import scipy.io
import random
import keras
from keras.datasets import mnist, cifar10, fashion_mnist


def get_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32')/255
    x_test = x_test.astype('float32')/255
    return (x_train, y_train), (x_test, y_test)


def get_cifar10():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    x_train = x_train.astype('float32')/255
    x_test = x_test.astype('float32')/255
    return (x_train, y_train), (x_test, y_test)


def binarize_mnist_class(_trainY, _testY):
    trainY = np.ones(len(_trainY), dtype=np.int32).reshape(len(_trainY), 1)
    trainY[_trainY % 2 == 1] = -1
    testY = np.ones(len(_testY), dtype=np.int32).reshape(len(_testY), 1)
    testY[_testY % 2 == 1] = -1
    return trainY, testY


def binarize_cifar_class(_trainY, _testY):
    trainY = -np.ones(len(_trainY), dtype=np.int32).reshape(len(_trainY), 1)
    trainY[(_trainY == 2) | (_trainY == 3) | (_trainY == 4) | (_trainY == 5) | (_trainY == 6) | (_trainY == 7)] = 1
    testY = -np.ones(len(_testY), dtype=np.int32).reshape(len(_testY), 1)
    testY[(_testY == 2) | (_testY == 3) | (_testY == 4) | (_testY == 5) | (_testY == 6) | (_testY == 7)] = 1
    return trainY, testY


def make_dataset(dataset, unlabeled_1, unlabeled_2, theta1, theta2, mode):
    def make_test_dataset(x, y):
        labels = np.unique(y)
        positive, negative = labels[1], labels[0]
        X, Y = np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.int32)
        n_p = (Y == positive).sum()
        n_n = (Y == negative).sum()
        prior_test = float(n_p) / (float(n_n) + float(n_p))
        Y_index = Y.reshape(X.shape[0], )
        Xp = X[Y_index == positive][:n_p]
        Xn = X[Y_index == negative][:n_n]
        X = np.asarray(np.concatenate((Xp, Xn)), dtype=np.float32)
        Y = np.concatenate((np.ones((n_p, 1), dtype=np.int32),
                            -np.ones((n_n, 1), dtype=np.int32)), axis=0)
        assert (len(X) == len(Y))
        perm = np.random.permutation(len(Y))
        X, Y = X[perm], Y[perm]
        return X, Y, prior_test

    def make_training_dataset(x, y, unlabeled1, unlabeled2, theta1, theta2, mode):
        labels = np.unique(y)
        positive, negative = labels[1], labels[0]
        X, Y = np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.int32)
        assert(len(X) == len(Y))
        perm = np.random.permutation(len(X))
        X, Y = X[perm], Y[perm]

        n1 = unlabeled1
        n2 = unlabeled2
        n1_p = int(n1 * theta1)
        n1_n = int(n1 - n1_p)
        n2_p = int(n2 * theta2)
        n2_n = int(n2 - n2_p)

        prior_UU = (float(n1)) / (float(n1) + float(n2))

        if mode == "UU" or mode == "UU_simp":
            # UU or UU_simp
            Y_index = Y.reshape(X.shape[0], )
            X1p = X[Y_index == positive][:n1_p]
            X1n = X[Y_index == negative][:n1_n]
            X2p = np.concatenate((X[Y_index == positive][n1_p:], X1p), axis=0)[:n2_p]
            X2n = np.concatenate((X[Y_index == negative][n1_n:], X1n), axis=0)[:n2_n]
            X1 = np.asarray(np.concatenate((X1p, X1n), axis=0), dtype=np.float32)
            X2 = np.asarray(np.concatenate((X2p, X2n), axis=0), dtype=np.float32)
        elif mode == "PN" or mode == "small_PN":
            # PN or small_PN
            Y_index = Y.reshape(X.shape[0], )
            P_data = X[Y_index == positive]
            N_data = X[Y_index == negative]
            X1p = P_data[:n1_p]
            X2n = N_data[:n2_n]
            X1 = np.asarray(X1p, dtype=np.float32)
            X2 = np.asarray(X2n, dtype=np.float32)
        else:
            raise ValueError("Mode name {} is unknown.".format(mode))
        np.random.shuffle(X1)
        np.random.shuffle(X2)
        return X1, X2, prior_UU

    (_trainX, _trainY), (_testX, _testY) = dataset
    U_data1, U_data2, prior_tr = make_training_dataset(_trainX, _trainY, unlabeled_1, unlabeled_2, theta1, theta2, mode)
    X, Y, prior_te = make_test_dataset(_testX, _testY)
    return U_data1, U_data2, prior_tr, X, Y, prior_te


def load_dataset(dataset_name, unlabeled1, unlabeled2, theta1, theta2, mode):
    if dataset_name == "mnist":
        (trainX, trainY), (testX, testY) = get_mnist()
        trainY, testY = binarize_mnist_class(trainY, testY)

    elif dataset_name == "cifar10":
        (trainX, trainY), (testX, testY) = get_cifar10()
        trainY, testY = binarize_cifar_class(trainY, testY)

    else:
        raise ValueError("Dataset name {} is unknown.".format(dataset_name))

    U_1, U_2, prior_tr, x_test, y_test, prior_te = make_dataset(((trainX, trainY), (testX, testY)),
                                                                    unlabeled1, unlabeled2, theta1, theta2, mode)
    return U_1, U_2, prior_tr, x_test, y_test, prior_te


