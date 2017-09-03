# coding: utf-8

# ダウンロード元のサイト：http://www.ffothello.org/informatique/la-base-wthor
# 碁盤が全部埋まる前に途中で終了しているような変なデータは入っていない
# 先行：黒
# states[i] -> st, actions[i] -> at に対応
# ランダムに並び替え済

import numpy as np
import urllib.request
import os

# サイトのファイルが更新されたときはサイトを見て以下の数字を変えること！
GAME_SUM = 120557  # 総対局数→120557対局×60手＝7233420手(約720万)
LATEST_YEAR = 2017  # 最新の年（1977年～LATEST_YEAR年のデータを使う）


# データのダウンロード
def download():
    os.mkdir('data')
    for year in range(LATEST_YEAR - 1977 + 1):
        urllib.request.urlretrieve('http://www.ffothello.org/wthor/base/WTH_{}.wtb'.format(1977 + year), 'data/WTH_{}.wtb'.format(1977 + year))
        print('{} download succeed'.format(1977 + year))


# データの読み込み・加工・保存
def load_and_save():
    # 碁盤の初期状態
    # y, x
    # 0:なし, 1:黒, 2:白
    initial_state = np.zeros([8, 8], dtype=np.int8)
    initial_state[4, 3] = 1  # 次に打つ人(=先行)の色は黒
    initial_state[3, 4] = 1
    initial_state[3, 3] = 2
    initial_state[4, 4] = 2

    # 碁盤の状態
    # y, x
    # 0:なし, 1:黒, 2:白
    now_state = initial_state.copy()

    # index, y, x
    # 0:なし, 1:次に打つ人の色, 2:反対の色
    states = np.zeros([GAME_SUM * 60, 8, 8], dtype=np.int8)

    # index
    # 打った所の番号
    actions = np.zeros(GAME_SUM * 60, dtype=np.int8)

    color = 1  # 1:黒, 2:白
    counter = 0
    random_index_list = np.random.choice(GAME_SUM * 60, GAME_SUM * 60,
                                         replace=False)  # 0以上GAME_SUM*60-1以下の重複なしの乱数：indexに使う

    for year in range(LATEST_YEAR - 1977 + 1):
        print(1977 + year)
        with open('data/WTH_{}.wtb'.format(1977 + year), 'br') as file:
            file.read(16)
            while True:
                head_data = file.read(8)
                if not head_data:  # .wtbでは最後は必ず本データ部分で終わっているからこれでよい
                    break
                raw_data = file.read(60)
                now_state = initial_state.copy()
                states[random_index_list[counter]] = now_state.copy()

                for (i, tmp) in enumerate(raw_data):
                    x = tmp // 10 - 1
                    y = tmp % 10 - 1

                    for j in range(2):
                        # 8方向捜索
                        is_changed = False  # ひっくり返されたかどうか
                        for k in range(3):  # k - 1 = -1, 0, 1となり移動方向が表せる
                            for l in range(3):
                                if k - 1 == 0 and l - 1 == 0:  # 両方0だと移動しない
                                    continue
                                for m in range(1, 8):
                                    xt = x + m * (k - 1)
                                    yt = y + m * (l - 1)
                                    if not (0 <= xt <= 7 and 0 <= yt <= 7):
                                        break
                                    elif now_state[yt, xt] == 0:
                                        break
                                    elif now_state[yt, xt] == color:
                                        if m > 1:
                                            for n in range(0, m):
                                                xt_ = x + n * (k - 1)
                                                yt_ = y + n * (l - 1)
                                                now_state[yt_, xt_] = color
                                            actions[random_index_list[counter]] = y * 8 + x
                                            is_changed = True
                                        break
                        if color == 1:
                            color = 2
                        else:
                            color = 1
                        if is_changed:
                            break

                    if i < 60 - 1:  # 最後の状態はstatesに使わない
                        states[random_index_list[counter + 1]] = now_state.copy()
                        if color == 2:  # 白が自分の手の時は1と2をひっくり返す
                            for j in range(8):
                                for k in range(8):
                                    if states[random_index_list[counter + 1], j, k] == 1:
                                        states[random_index_list[counter + 1], j, k] = 2
                                    elif states[random_index_list[counter + 1], j, k] == 2:
                                        states[random_index_list[counter + 1], j, k] = 1

                    counter += 1

        print('{} load succeed'.format(1977 + year))

    # 保存
    np.save('states.npy', states)
    np.save('actions.npy', actions)

    print('save succeed')

    return states, actions


if __name__ == '__main__':
    download()
    states, actions = load_and_save()
