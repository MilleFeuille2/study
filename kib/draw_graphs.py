# -*- encoding:utf-8 -*-
import os
import sys
from argparse import ArgumentParser
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


# 軸のパーセント表示のため
fmt = '%.0f%%'
xticks = mtick.FormatStrFormatter(fmt)

# 引数を取得する
parser = ArgumentParser(description='File name')
parser.add_argument('-filename', type=str, default='result.csv', help='input file name')  # 解析結果のファイル名
parser.add_argument('-outpath', type=str, default='.', help='relative output path')  # グラフをアウトプットする相対パス
args = parser.parse_args()

infile = os.path.join(os.getcwd(), args.filename)
outpath = os.path.join(os.getcwd(), args.outpath)


def get_data(file_path):
    """
    解析結果を読み取る
    """
    df = pd.read_csv(file_path, encoding='SHIFT-JIS')

    return df


def draw_score(df_data, name='chara'):
    """
    スコア分布を描画する
    """
    score_t = df_data[df_data['y'] == 1]
    score_f = df_data[df_data['y'] == 0]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(score_f.index, score_f['score'], color='blue', label='False')
    ax.scatter(score_t.index, score_t['score'], color='red', label='True')
    ax.set_ylim([0, 10000])
    ax.set_xlabel('閲覧数（スコアの降順）', fontsize=14)
    ax.set_ylabel('スコア', fontsize=14)
    ax.legend(loc=1, fontsize=14)
    ax.set_title('スコア分布 ({0})'.format(name), fontsize=16)
    plt.savefig('score_distribution_{0}.png'.format(name))
    plt.clf()


def draw_recall(df_data, name='chara'):
    """
    Recall Rateを描画する
    """
    df_data['x'] = df_data.index / (len(df_data) - 1)

    count_t = (df_data['y_'] == 1).sum()
    df_data['ideal'] = [i / count_t if i < count_t else 1 for i in range(len(df_data))]
    df_data['kibit'] = [(df_data.loc[:i, 'y_'] == 1).sum() / count_t for i in range(len(df_data))]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(df_data['x'], df_data['ideal'], color='blue', label='ideal')
    ax.plot(df_data['x'], df_data['kibit'], color='red', label='kibit')
    ax.plot([0, 1], [0, 1], color='green', label='random')

    ax.set_xlabel('閲覧率', fontsize=14)
    ax.set_xticks(np.linspace(0, 1, 11))
    ax.set_xticklabels(['{0}%'.format(int(x*100)) for x in ax.get_xticks()], fontsize=12)

    ax.set_ylabel('抽出率', fontsize=14)
    ax.set_yticks(np.linspace(0, 1, 11))
    ax.set_yticklabels(['{0}%'.format(int(x*100)) for x in ax.get_yticks()], fontsize=12)

    ax.legend(loc=4, fontsize=14)
    ax.set_title('Recall Rate ({0})'.format(name), fontsize=18)
    plt.savefig('recall_rate_{0}.png'.format(name))
    plt.clf()


def draw_precision(df_data, name='chara'):
    """
    Precision Rateを描画する
    """
    df_data['x'] = (df_data.index / (len(df_data) - 1))

    df_data['kibit'] = [(df_data.loc[:i, 'y_'] == 1).sum() / (i + 1) for i in range(len(df_data))]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(df_data['x'], df_data['kibit'], color='blue')

    ax.set_xlabel('閲覧率', fontsize=14)
    ax.set_xticks(np.linspace(0, 1, 11))
    ax.set_xticklabels(['{0}%'.format(int(x*100)) for x in ax.get_xticks()], fontsize=12)

    ax.set_ylabel('正解率', fontsize=14)
    ax.set_yticks(np.linspace(0, 1, 11))
    ax.set_yticklabels(['{0}%'.format(int(x*100)) for x in ax.get_yticks()], fontsize=12)
    ax.set_ylim([0.0, 1.0])

    ax.set_title('Precision Rate ({0})'.format(name), fontsize=16)
    plt.savefig('precision_rate_{0}.png'.format(name))
    plt.clf()


if __name__ == '__main__':

    # 分析結果を読み込む
    df_data = get_data(infile)

    # カラム名『スコア』があれば特性分析のグラフを描く
    if 'スコア' in df_data.columns:

        df_data = df_data.rename(columns={'スコア': 'score'})

        if not os.path.exists(outpath):
            os.mkdir(outpath)
        os.chdir(outpath)

        # スコアの降順に並び替え、教師データ列を指定する
        df_data = df_data.sort_values(by=['score'], ascending=False).reset_index(drop=True)
        df_data['y_'] = df_data['y']

        # スコア分布を描く
        draw_score(df_data)

        # Recall Rateを描く
        draw_recall(df_data)

        # Precision Rateを描く
        draw_precision(df_data)

    # カラム名『軸1_スコア』があれば交差検証のグラフを描く
    if '軸1_スコア' in df_data.columns:

        for i in range(1, 5):

            # スコアの降順に並び替え、教師データ列を指定する
            df_data = df_data.rename(columns={'軸{0}_スコア'.format(str(i)): 'score{0}'.format(i)})
            df_data = df_data.sort_values(by=['score{0}'.format(str(i))], ascending=False).reset_index(drop=True)
            df_data['y_'] = df_data['y{0}'.format(str(i))]

            # Recall Rateを描く
            draw_recall(df_data, str(i))

            # Precision Rateを描く
            draw_precision(df_data, str(i))

    print('finish!')

