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
    def __init__(self, data, decomposer=None):
        self.original_data = data

        self.decomposited = False
        if decomposer:
            self.decomposited = True
            self.data = decomposer.fit(data)

    def __next__(self):
        pass

    def winner_takes_all(self):
        pass

    def func_neighborhood(self):
        pass

    def update(self):
        pass


class Drawer(object):
    def __init__(self, width, height):
        self.width = width
        self.height = height

        self.fig = plt.figure()

    def print(self, dataset):
        """
        :param data: [png, (point)]
        :return:
        """
        for data in dataset:
            img, pos = data
            subfig = self.fig.add_subplot(self.height, self.width, 10 * pos[0] + pos[1] + 1)
            subfig.get_xaxis().set_visible(False)
            subfig.get_yaxis().set_visible(False)
            plt.imshow(img)

        plt.show()


if __name__ == '__main__':
    dataset = mnist.get_mnist(withlabel=False, ndim=2)[0][:10000]

    sample = dataset[:100]
    sample_data = []

    for i in range(10):
        for j in range(10):
            sample_data.append([sample[10 * i + j], (i, j)])

    drawer = Drawer(10, 10)
    drawer.print(sample_data)
