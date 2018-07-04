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
    def __init__(self, dim, data, epoch=1, decompose=None):
        self.dim = dim
        self.original_data = data

        if decompose:
            self.data = decompose.fit(data)
        else:
            self.data = data

        data_size = len(self.data[0])
        self.node = np.random.normal(size=(dim ** 2, data_size))
        self.winner = None

        self.targetId = 0
        self.epoch = epoch
        self.rate = 1.0

    def __iter__(self):
        return self

    def __next__(self):
        target_neuron = self.data[self.targetId]
        self.set_winner(target_neuron)
        h = self.func_neighborhood()
        self.update(target_neuron, h)

        if self.targetId == len(self.data) - 1:
            self.targetId = 0
            self.epoch -= 1

            if self.epoch == 0:
                raise StopIteration()

        else:
            self.targetId += 1
        self.rate *= 0.9

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
        return np.argmin(ref), np.min(ref)

    def set_winner(self, neuron):
        self.winner, _ = self.nearliest_node(neuron)

    def func_neighborhood(self):
        wi = self.winner // self.dim
        wj = self.winner % self.dim
        dists = np.zeros((self.dim ** 2, 1))
        for i in range(self.dim):
            for j in range(self.dim):
                dists[i * self.dim + j] = (wi - i) ** 2 + (wj - j) ** 2
        return np.exp(-dists / float(self.dim))

    def update(self, src, h):
        self.node = self.node + self.rate * h * (src - self.node)


if __name__ == '__main__':
    map_dim = 20

    print("データセットの用意")
    dataset = mnist.get_mnist(withlabel=False, ndim=1)[0][:20000]

    print("SOMの準備")
    som = SelfOrganizationMap(map_dim, dataset, 2, PCA(dim=50))

    print("イタレータ実行")
    for i, sample in enumerate(som):
        if i % 4000 != 0:
            continue

        sample = sample.reshape(map_dim, map_dim, 28, 28)
        sample = sample.transpose(0, 2, 1, 3).reshape((map_dim * 28, map_dim * 28))
        plt.imshow(sample)
        plt.savefig("sample_{}".format(i))
