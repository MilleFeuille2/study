# -*- encoding:utf-8 -*-
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 引数


def get_args():

    parser = argparse.ArgumentParser()

    # フォルダ名
    # ファイル名
    # 特性分析か交差検証か

    return None


def get_data():

    df = pd.read_csv('./test.csv', encoding='utf-8')

    return df


def draw_score(df_data, name):

    # スコアの降順に並べる
    df_data = df_data.sort_values(by=['score'], ascending=False).reset_index(drop=True)
    
    score_t = df_data[df_data['y'] == 1]
    score_f = df_data[df_data['y'] == 0]

    plt.scatter(score_t.index, score_t['score'], color='red', label='true')
    plt.scatter(score_f.index, score_f['score'], color='blue', label='false')
    plt.xlabel('rank (descending)')
    plt.ylabel('score')
    plt.legend(loc=1)
    plt.title('score distribution ({0})'.format(name))
    plt.savefig('./score_distribution_{0}.png'.format(name))
    plt.clf()


def draw_recall(df_data, name):

    # スコアの降順に並べる
    if name == 'all':
        df_data = df_data.sort_values(by=['score'], ascending=False).reset_index(drop=True)
    else:
        df_data = df_data.sort_values(by=['score{0}'.format(name)], ascending=False).reset_index(drop=True)

    count_t = (df_data['y'] == 1).sum()

    df_data['x'] = df_data.index / (len(df_data) - 1)

    # 本来のRecall Rateを求める
    df_data['ideal'] = [i / count_t if i < count_t else 1 for i in range(len(df_data))]
    df_data['kibit'] = [(df_data.loc[:i, 'y'] == 1).sum() / count_t for i in range(len(df_data))]

    plt.plot(df_data['x'], df_data['ideal'], color='blue', label='ideal')
    plt.plot(df_data['x'], df_data['kibit'], color='red', label='kibit')
    plt.plot([0, 1], [0, 1], color='green', label='random')
    plt.xlabel('access rate')
    plt.ylabel('extraction rate')
    plt.legend(loc=4)
    plt.title('recall rate {0}'.format(name))
    plt.savefig('./recall_rate_{0}.png'.format(name))
    plt.clf()


def draw_precision(df_data, name):

    df_data = df_data.sort_values(by=['score'], ascending=False).reset_index(drop=True)

    count_t = (df_data['y'] == 1).sum()

    df_data['x'] = (df_data.index / (len(df_data) - 1))

    df_data['kibit'] = [(df_data.loc[:i, 'y'] == 1).sum() / (i + 1) for i in range(len(df_data))]

    plt.plot(df_data['x'], df_data['kibit'], color='blue')
    plt.ylim([0.0, 1.0])
    plt.xlabel('access rate')
    plt.ylabel('precision rate')
    plt.title('precision rate {0}'.format(name))
    plt.savefig('./precision_rate_{0}.png'.format(name))
    plt.clf()


def perc(x):
    return x.map('{:.2%}'.format)


if __name__ == '__main__':

    # 引数を取得する
    args = get_args()

    # 分析結果を読み込む
    df_data = get_data()

    # カラム名『スコア』があれば特性分析のグラフを描く
    if 'スコア' in df_data.columns:

        df_data = df_data.rename(columns={'スコア': 'score'})

        # スコア分布を描く
        draw_score(df_data, 'all')

        # Recall Rateを描く
        draw_recall(df_data, 'all')

        # Precision Rateを描く
        draw_precision(df_data, 'all')

    # カラム名『軸1_スコア』があれば交差検証のグラフを描く
    if '軸1_スコア' in df_data.columns:

        for i in range(4):

            df_data = df_data.rename(columns={'軸{0}_スコア'.format(str(i)): 'score{0}'.format(i)})

            # Recall Rateを描く
            draw_recall(df_data, str(i))

            # Precision Rateを描く
            draw_precision(df_data, str(i))

    print('finish!')

