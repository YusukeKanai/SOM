"""
__author__ = "Yusuke Kanai"
__copyright__ = "Copyright (C) 2018 Yusuke Kanai"
__licence__ = "MIT"
__version = "0.1"
"""

# -*- coding:utf-8 -*-

from __future__ import print_function, unicode_literals, division
from sklearn import decomposition
import matplotlib.pyplot as plt
from chainer.datasets import mnist
import numpy as np


class PCA(object):
    """
    主成分分析で次元圧縮
    """

    def __init__(self, dim):
        """
        :param dim: 次元数
        """
        self.dim = dim
        self.pca = decomposition.PCA(n_components=self.dim)

    def fit(self, data):
        """
        :param data: 元データ
        :return: 圧縮後のデータ
        """
        self.pca.fit(data)
        return self.pca.transform(data)


class SelfOrganizationMap(object):
    def __init__(self, dim, data, epoch=1, decomposer=None):
        self.dim = dim
        self.original_data = data

        self.decomposited = False
        if decomposer:
            self.decomposited = True
            self.data = decomposer.fit(data)
        else:
            self.data = data

        dim_size = len(self.data[0])
        self.node = np.random.rand(dim * dim, dim_size)
        self.winner = None

        self.targetId = 0
        self.epoch = epoch

    def __iter__(self):
        return self

    def __next__(self):
        if self.epoch == 0:
            raise StopIteration()

        target_neuron = self.data[self.targetId]
        self.set_winner(target_neuron)
        h = self.func_neighborhood()
        self.update(h)

        if self.targetId == len(self.data) - 1:
            self.targetId = 0
            self.epoch -= 1

            if self.epoch == 0:
                raise StopIteration()

        else:
            self.targetId += 1

        # get nearliest image
        image_node_map = dict()
        for i in range(len(self.data)):
            nid, ref = self.nearliest_node(self.data[i])
            if nid in image_node_map:
                _, ref2 = image_node_map[nid]
                if ref2 < ref:
                    continue
            image_node_map[nid] = (i, ref)

        # set image to node
        res = np.zeros((self.dim ** 2, 784))
        for k, v in image_node_map.items():
            res[k] = self.original_data[v[0]]

        return res

    def nearliest_node(self, src):
        z = self.node - src
        ref = np.sum(z ** 2, axis=1)
        return np.argmax(ref), np.max(ref)

    def set_winner(self, neuron):
        self.winner, _ = self.nearliest_node(neuron)

    def func_neighborhood(self):
        dists = (self.node - self.node[self.winner]) ** 2
        return np.exp(-dists / (self.dim ** 2))

    def update(self, h):
        self.node = self.node + 0.05 * h * self.data - self.node[self.winner]


if __name__ == '__main__':

    print("データセットの用意")
    dataset = mnist.get_mnist(withlabel=False, ndim=1)[0][:10000]

    sample = dataset[:100]

    print("SOMの準備")

    som = SelfOrganizationMap(10, sample, 1, PCA(dim=20))

    print("イタレータ実行")
    for sample in som:
        sample = sample.reshape(10, 10, 28, 28)
        sample = sample.transpose(0, 2, 1, 3).reshape((10 * 28, 10 * 28))
        plt.imshow(sample)
        plt.show()
