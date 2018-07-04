"""
__author__ = "Yusuke Kanai"
__copyright__ = "Copyright (C) 2018 Yusuke Kanai"
__licence__ = "MIT"
__version = "0.1"
"""

# -*- coding:utf-8 -*-

from __future__ import print_function, unicode_literals, division
from sklearn import decomposition


class PCA(object):
    def __init__(self, dim):
        self.dim = dim
        self.pca = decomposition.PCA(n_components=self.dim)

    def fit(self, data):
        self.pca.fit(data)
        return self.pca.transform(data)


class SelfOrganizationMap(object):
    def __init__(self, data, decomposer=None):
        pass

    def __next__(self):
        pass

    def winner_takes_all(self):
        pass

    def func_neighborhood(self):
        pass

    def update(self):
        pass


class Drawer(object):
    def __init__(self):
        pass


if __name__ == '__main__':
    pass
