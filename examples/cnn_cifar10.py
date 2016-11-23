#!/usr/bin/env python
# coding: utf-8
import cPickle
import time
import numpy as np
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], ".."))
from nnet import nnet

def load(folder):
    #files = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5"]
    files = ["data_batch_1"]
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    data = []
    labels = []
    for data_file in files:
        with open(os.path.join(cur_dir, folder, data_file)) as fo:
            images = cPickle.load(fo)
            total = len(images["labels"])
            for i in range(0, total):
                one_data = images["data"][i]
                one_label = images["labels"][i]
                data.append(np.reshape(one_data, (3, 32, 32)).T)
                labels.append(one_label)
    return data, labels

def run():
    # Fetch data
    # data, labels = load("../data")
    print(nnet)

    """
    mnist = sklearn.datasets.fetch_mldata('MNIST original', data_home='./data')
    split = 60000
    X_train = np.reshape(mnist.data[:split], (-1, 1, 28, 28))/255.0
    y_train = mnist.target[:split]
    X_test = np.reshape(mnist.data[split:], (-1, 1, 28, 28))/255.0
    y_test = mnist.target[split:]
    n_classes = np.unique(y_train).size

    # Downsample training data
    n_train_samples = 3000
    train_idxs = np.random.random_integers(0, split-1, n_train_samples)
    X_train = X_train[train_idxs, ...]
    y_train = y_train[train_idxs, ...]

    # Setup convolutional neural network
    nn = nnet.NeuralNetwork(
        layers=[
            nnet.Conv(
                n_feats=12,
                filter_shape=(5, 5),
                strides=(1, 1),
                weight_scale=0.1,
                weight_decay=0.001,
            ),
            nnet.Activation('relu'),
            nnet.Pool(
                pool_shape=(2, 2),
                strides=(2, 2),
                mode='max',
            ),
            nnet.Conv(
                n_feats=16,
                filter_shape=(5, 5),
                strides=(1, 1),
                weight_scale=0.1,
                weight_decay=0.001,
            ),
            nnet.Activation('relu'),
            nnet.Flatten(),
            nnet.Linear(
                n_out=n_classes,
                weight_scale=0.1,
                weight_decay=0.02,
            ),
            nnet.LogRegression(),
        ],
    )

    # Train neural network
    t0 = time.time()
    nn.fit(X_train, y_train, learning_rate=0.05, max_iter=3, batch_size=32)
    t1 = time.time()
    print('Duration: %.1fs' % (t1-t0))

    # Evaluate on test data
    error = nn.error(X_test, y_test)
    print('Test error rate: %.4f' % error)
    """

if __name__ == '__main__':
    run()
