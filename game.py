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


# 1�^�[���V�~�����[�g
# position  y, x
# color  1:��, 2:��
# return  True:�L��, False:����
def simulation(state, position, color):
    y = position[0]
    x = position[1]

    # ���łɐ΂��u���Ă���ꍇ�����O
    if state[y, x] != 0:
        return False

    # 8�����{��
    is_changed = False  # �Ђ�����Ԃ��ꂽ���ǂ���
    for i in range(3):  # i - 1 = -1, 0, 1�ƂȂ�ړ��������\����
        for j in range(3):
            if i - 1 == 0 and j - 1 == 0:  # ����0���ƈړ����Ȃ�
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


# �p�X�m�F
# color  1:��, 2:��
# return  True:�p�X, False:�p�X�Ȃ�
def is_pass(state, color):
    for i in range(8):
        for j in range(8):
            if simulation(state.copy(), (i, j), color):
                return False
    return True


# ��Ղ̏�Ԃ�\��
def show(state):
    print(' ', end='')

    for i in range(8):
        print(i, end='')
    print() # ���s

    for i in range(8):
        print(i, end='')
        for j in range(8):
            if state[i][j] == 0:
                print('*', end='')
            elif state[i][j] == 1:
                print('b', end='')
            else:
                print('w', end='')
        print()  # ���s


model = L.Classifier(network.AgentNet(), lossfun=softmax_cross_entropy)
serializers.load_npz('model.npz', model)

# ��Ղ̏��
# y, x
# 0:�Ȃ�, 1:��, 2:���@���@�R���s���[�^�[����1�ɂȂ�悤��
state = np.zeros([8, 8], dtype=np.int8)
state[4, 3] = 1
state[3, 4] = 1
state[3, 3] = 2
state[4, 4] = 2

print('You are white, second.')  # �R���s���[�^�[����1�ɂȂ�悤��
print('input : y,x')
print()  # ���s

former_is_pass = False  # �����p�X�ŏI���̂��߁A�O��p�X���������L�^

show(state)  # ��Ղ̏�Ԃ�\��
print()  # ���s

while True:
    # �R���s���[�^�[��
    if is_pass(state, 1):
        if former_is_pass:
            break
        else:
            former_is_pass = True
    else:
        former_is_pass = False

        state_var = chainer.Variable(state.reshape(1, 1, 8, 8).astype(np.float32))
        action_probabilities = model.predictor(state_var).data.reshape(64)

        # �m�����ɍs������ׂ�
        # �K���ɏ������̂ŃA���S���Y���̌����������Ǝv��
        action_list = [0]  # ���X�g
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
        print()  # ���s
        print()  # ���s

        show(state)  # ��Ղ̏�Ԃ�\��
        print()  # ���s

    # ���[�U�[��
    if is_pass(state, 2):
        if former_is_pass:
            break
        else:
            former_is_pass = True
    else:
        former_is_pass = False

        while True:
            position = [int(e) for e in input().split(',')]  # ���[�U�[����̓���
            if simulation(state, position, 2):
                break
            else:
                print('Invalid Position')
        print()  # ���s

        show(state)  # ��Ղ̏�Ԃ�\��
        print()  # ���s


# �W�v
black_score = 0
white_score = 0
for i in range(8):
    for j in range(8):
        if state[i][j] == 1:
            black_score += 1
        elif state[i][j] == 2:
            white_score += 1


# ���s�\��
if black_score == white_score:
    print('Draw.  b(computer) - w(you) :', black_score, '-', white_score)
elif black_score > white_score:
    print('Computer Win.  b(computer) - w(you) :', black_score, '-', white_score)
else:
    print('You Win!  b(computer) - w(you) :', black_score, '-', white_score)