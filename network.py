# coding: utf-8

import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L


FILTERS_NUM = 50  # フィルター数(k)
HIDDEN_LAYER_NUM = 10  # 隠れ層の数(m-2)


class AgentNet(Chain):
    def __init__(self):
        super(AgentNet, self).__init__()
        net = [('conv1', L.Convolution2D(1, FILTERS_NUM, 3, 1, 1, nobias=True))]
        net += [('bn1', L.BatchNormalization(FILTERS_NUM))]
        net += [('_relu1', F.ReLU())]
        for i in range(HIDDEN_LAYER_NUM - 2):
            net += [('conv{}'.format(i + 2), L.Convolution2D(FILTERS_NUM, FILTERS_NUM, 3, 1, 1, nobias=True))]
            net += [('bn{}'.format(i + 2), L.BatchNormalization(FILTERS_NUM))]
            net += [('_relu{}'.format(i + 2), F.ReLU())]
        net += [('conv{}'.format(HIDDEN_LAYER_NUM), L.Convolution2D(FILTERS_NUM, 1, 1, 1, 0, nobias=False))]
        with self.init_scope():
            for n in net:
                if not n[0].startswith('_'):
                    setattr(self, n[0], n[1])
        self.forward = net

    def __call__(self, x):
        size = x.data.shape[0]
        for n, f in self.forward:
            if not n.startswith('_'):
                x = getattr(self, n)(x)
            else:
                x = f(x)
        x = F.reshape(x, (size, 64))  # 出力を8×8→64に展開
        if chainer.config.train:  # 訓練時：softmax_cross_entropy関数を使うのでsoftmax関数は省く
            return x
        return F.softmax(x)