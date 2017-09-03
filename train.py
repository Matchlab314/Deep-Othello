# coding: utf-8

import load
import network
import numpy as np
import os
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.functions.loss.softmax_cross_entropy import softmax_cross_entropy


TEST_DATA_SIZE = 100000  # テストデータのサイズ
MINIBATCH_SIZE = 100  # ミニバッチサイズ
EVALUATION_SIZE = 1000  # 評価のときのデータサイズ


# データの読み込み・加工
if os.path.isfile('states.npy') and os.path.isfile('actions.npy'):
    states = np.load('states.npy')
    actions = np.load('actions.npy')
else:
    download()  # ファイルダウンロード
    states, actions = load.load_and_save()  # データの読み込み・加工・保存

test_x = states[:TEST_DATA_SIZE].copy()  # ランダムに並び替え済み
train_x = states[TEST_DATA_SIZE:].copy()
del states  # メモリがもったいないので強制解放
test_y = actions[:TEST_DATA_SIZE].copy()
train_y = actions[TEST_DATA_SIZE:].copy()
del actions


model = L.Classifier(network.AgentNet(), lossfun=softmax_cross_entropy)
if os.path.isfile('model.npz'):  # モデル読み込み
    serializers.load_npz('model.npz', model)

optimizer = optimizers.Adam()
optimizer.setup(model)

# 学習ループ
for epoch in range(100):
    for i in range(100):
        index = np.random.choice(train_x.shape[0], MINIBATCH_SIZE, replace=False)
        x = chainer.Variable(train_x[index].reshape(MINIBATCH_SIZE, 1, 8, 8).astype(np.float32))
        t = chainer.Variable(train_y[index].astype(np.int32))
        optimizer.update(model, x, t)

    # 評価
    index = np.random.choice(test_x.shape[0], EVALUATION_SIZE, replace=False)
    x = chainer.Variable(test_x[index].reshape(EVALUATION_SIZE, 1, 8, 8).astype(np.float32))
    t = chainer.Variable(test_y[index].astype(np.int32))
    print('epoch :', epoch, '  loss :', model(x, t).data)

    serializers.save_npz('model.npz', model)  # モデル保存
