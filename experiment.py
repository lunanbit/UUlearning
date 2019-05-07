from __future__ import print_function
import sys
import os
import keras
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.optimizers import SGD, Adam
from tensorflow import set_random_seed
from helper import plot_loss, lr_decay_schedule
from models import MultiLayerPerceptron, Resnet32Model
import argparse
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def get_args():
    parser = argparse.ArgumentParser(
        description='UU learning Keras implementation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--batchsize', type=int, default=128)
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'cifar10'])
    parser.add_argument('--mode', type=str, default='UU', choices=['UU', 'PN', 'UU_simp', 'small_PN'])
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--loss', type=str, default="sigmoid", choices=['sigmoid', 'logistic'])
    parser.add_argument('--model', type=str, default='mlp', choices=['resnet32', 'mlp'])
    parser.add_argument('--learningrate', type=float, default=1e-3)
    parser.add_argument('--weightdecay', type=float, default=1e-4)
    parser.add_argument('--unlabeled1', type=int, default=30000)
    parser.add_argument('--unlabeled2', type=int, default=30000)
    parser.add_argument('--theta1', type=float, default=0.9, choices=[0.9, 0.8])
    parser.add_argument('--theta2', type=float, default=0.1, choices=[0.1, 0.2])
    args = parser.parse_args()

    if args.dataset == "mnist":
        if args.mode == "small_PN":
            args.unlabeled1 = 3000
            args.unlabeled2 = 3000
        else:
            args.unlabeled1 = 30000
            args.unlabeled2 = 30000

        if args.loss == 'sigmoid':
            args.learningrate = 1e-3
        else:
            args.learningrate = 1e-4

        args.batchsize = 128
        args.weightdecay = 1e-4
        args.model = "mlp"

    elif args.dataset == "cifar10":
        if args.mode == "UU" or args.mode == "UU_simp":
            args.unlabeled1 = 25000
            args.unlabeled2 = 25000
        elif args.mode == "PN":
            args.unlabeled1 = 30000
            args.unlabeled2 = 20000
        else:
            args.unlabeled1 = 3000
            args.unlabeled2 = 2000

        if args.loss == 'sigmoid':
            args.learningrate = 1e-5
        elif args.loss == 'logistic':
            args.learningrate = 1e-6

        args.batchsize = 500
        args.weightdecay = 5e-3
        args.model = "resnet32"

    if args.mode == "UU" or args.mode == "UU_simp":
        assert (0 < args.theta1 <= 1)
        assert (0 < args.theta1 <= 1)
        assert (args.theta1 > args.theta2)
        assert (1 - args.theta1 + args.theta2 < 0.5)
    elif args.mode == "PN" or args.mode == "small_PN":
        args.theta1 = 1
        args.theta2 = 0

    assert (args.batchsize > 0)
    assert (args.epoch > 0)

    if args.dataset == "mnist":
        assert (0 < args.unlabeled1 <= 60000)
        assert (0 < args.unlabeled2 <= 60000)
    else:
        assert (0 < args.unlabeled1 <= 50000)
        assert (0 < args.unlabeled2 <= 50000)

    args.lr_decay = lr_decay_schedule(args.loss, args.theta1, args.theta2)

    return args


def build_file_name(dataset, mode, nb_U1, nb_U2, theta1, theta2, loss_type, phase, figure):
    if figure:
        format = '.png'
    else:
        format = '.txt'
    return (os.path.dirname(os.path.realpath(__file__)) +
            '/output/' + dataset + '/' +
            mode + '_' +
            str(nb_U1) + '_' +
            str(nb_U2) + '_' +
            str(theta1) + '_' +
            str(theta2) + '_' +
            loss_type + '_' +
            phase + '_' + format)


def get_network(name, **kwargs):
    if name == 'resnet32':
        return Resnet32Model(**kwargs)
    if name == 'mlp':
        return MultiLayerPerceptron(**kwargs)
    else:
        raise NotImplementedError


def get_optimizer(name, **kwargs):
    if name == 'resnet32':
        return Adam(**kwargs)
    if name == 'mlp':
        return SGD(**kwargs)
    else:
        raise NotImplementedError


def exp(args):
    ExpModel = get_network(args.model,
                           dataset=args.dataset,
                           nb_U1=args.unlabeled1,
                           nb_U2=args.unlabeled2,
                           theta1=args.theta1,
                           theta2=args.theta2,
                           mode=args.mode,
                           loss_type=args.loss,
                           weight_decay=args.weightdecay)

    ExpModel.optimizer = get_optimizer(args.model, lr=args.learningrate, decay=args.lr_decay)

    print('Loading {} ...'.format(args.dataset))
    U_data1, U_data2, prior_true, x_test, y_test, prior = ExpModel.get_data()
    print('Done!')

    input_shape = U_data1.shape[1:]
    ExpModel.build_model(prior, input_shape)
    history, loss_test = ExpModel.fit_model(U_data1=U_data1,
                                            U_data2=U_data2,
                                            batch_size=args.batchsize,
                                            epochs=args.epoch,
                                            x_test=x_test,
                                            y_test=y_test)

    np_loss_test = np.array(loss_test)
    loss_test_file= build_file_name(args.dataset, args.mode, args.unlabeled1, args.unlabeled2, args.theta1, args.theta2, args.loss, phase='test', figure=False)
    np.savetxt(loss_test_file, np_loss_test, newline="\r\n")

    plot_loss(np_loss_test, args.epoch)
    figure_file = build_file_name(args.dataset, args.mode, args.unlabeled1, args.unlabeled2, args.theta1, args.theta2, args.loss, phase='test', figure=True)
    plt.savefig(figure_file)


if __name__ == '__main__':
    args = get_args()

    np.random.seed(10000)
    set_random_seed(10000)

    print("mode: {}".format(args.mode))
    print("loss: {}".format(args.loss))
    print("model: {}".format(args.model))
    print("unlabeled1: {}".format(args.unlabeled1))
    print("unlabeled2: {}".format(args.unlabeled2))

    exp(args)
