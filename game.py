# coding: utf-8

import network
import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.functions.loss.softmax_cross_entropy import softmax_cross_entropy


# 1ターンシミュレート
# position  y, x
# color  1:黒, 2:白
# return  True:有効, False:無効
def simulation(state, position, color):
    y = position[0]
    x = position[1]

    # すでに石が置いてある場合を除外
    if state[y, x] != 0:
        return False

    # 8方向捜索
    is_changed = False  # ひっくり返されたかどうか
    for i in range(3):  # i - 1 = -1, 0, 1となり移動方向が表せる
        for j in range(3):
            if i - 1 == 0 and j - 1 == 0:  # 両方0だと移動しない
                continue
            for k in range(1, 8):
                xt = x + k * (i - 1)
                yt = y + k * (j - 1)
                if not (0 <= xt <= 7 and 0 <= yt <= 7):
                    break
                elif state[yt, xt] == 0:
                    break
                elif state[yt, xt] == color:
                    if k > 1:
                        for l in range(0, k):
                            xt_ = x + l * (i - 1)
                            yt_ = y + l * (j - 1)
                            state[yt_, xt_] = color
                        is_changed = True
                    break

    return is_changed


# パス確認
# color  1:黒, 2:白
# return  True:パス, False:パスなし
def is_pass(state, color):
    for i in range(8):
        for j in range(8):
            if simulation(state.copy(), (i, j), color):
                return False
    return True


# 碁盤の状態を表示
def show(state):
    print(' ', end='')

    for i in range(8):
        print(i, end='')
    print() # 改行

    for i in range(8):
        print(i, end='')
        for j in range(8):
            if state[i][j] == 0:
                print('*', end='')
            elif state[i][j] == 1:
                print('b', end='')
            else:
                print('w', end='')
        print()  # 改行


model = L.Classifier(network.AgentNet(), lossfun=softmax_cross_entropy)
serializers.load_npz('model.npz', model)

# 碁盤の状態
# y, x
# 0:なし, 1:黒, 2:白　→　コンピューター側が1になるように
state = np.zeros([8, 8], dtype=np.int8)
state[4, 3] = 1
state[3, 4] = 1
state[3, 3] = 2
state[4, 4] = 2

print('You are white, second.')  # コンピューター側が1になるように
print('input : y,x')
print()  # 改行

former_is_pass = False  # 両方パスで終了のため、前回パスしたかを記録

show(state)  # 碁盤の状態を表示
print()  # 改行

while True:
    # コンピューター側
    if is_pass(state, 1):
        if former_is_pass:
            break
        else:
            former_is_pass = True
    else:
        former_is_pass = False

        state_var = chainer.Variable(state.reshape(1, 1, 8, 8).astype(np.float32))
        action_probabilities = model.predictor(state_var).data.reshape(64)

        # 確率順に行動を並べる
        # 適当に書いたのでアルゴリズムの効率が悪いと思う
        action_list = [0]  # リスト
        for i in range(1, 64):
            for j in range(i):
                if action_probabilities[i] > action_probabilities[action_list[j]]:
                    action_list.insert(j, i)
                    break
            else:
                action_list.append(i)

        for i in range(64):
            action = action_list[i]
            position = (action // 8, action % 8)
            print('{0},{1}'.format(position[0], position[1]), end=' ')
            if simulation(state, position, 1):
                break
        print()  # 改行
        print()  # 改行

        show(state)  # 碁盤の状態を表示
        print()  # 改行

    # ユーザー側
    if is_pass(state, 2):
        if former_is_pass:
            break
        else:
            former_is_pass = True
    else:
        former_is_pass = False

        while True:
            position = [int(e) for e in input().split(',')]  # ユーザーからの入力
            if simulation(state, position, 2):
                break
            else:
                print('Invalid Position')
        print()  # 改行

        show(state)  # 碁盤の状態を表示
        print()  # 改行


# 集計
black_score = 0
white_score = 0
for i in range(8):
    for j in range(8):
        if state[i][j] == 1:
            black_score += 1
        elif state[i][j] == 2:
            white_score += 1


# 勝敗表示
if black_score == white_score:
    print('Draw.  b(computer) - w(you) :', black_score, '-', white_score)
elif black_score > white_score:
    print('Computer Win.  b(computer) - w(you) :', black_score, '-', white_score)
else:
    print('You Win!  b(computer) - w(you) :', black_score, '-', white_score)