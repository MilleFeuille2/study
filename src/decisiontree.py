# -*- encoding:utf-8 -*-
import copy
import pandas as pd
import numpy as np
import pydotplus
from datetime import datetime
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split, GridSearchCV


def main():

    df_data = pd.read_csv('../data/adult/adult.csv')

    x = df_data.iloc[:, :-1]
    x = x[['age', 'fnlwgt', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week']]
    y = (df_data.iloc[:, -1] == '>50K').astype(int)

    # 通算/非通算それぞれを75%学習・25%評価に分割し、分割後結合する
    y_tol, y_notol = y[y == 1], y[y == 0]
    x_tol, x_notol = x.loc[y_tol.index, :], x.loc[y_notol.index, :]
    x_tol_train, x_tol_test, y_tol_train, y_tol_test = train_test_split(x_tol, y_tol, random_state=1)
    x_notol_train, x_notol_test, y_notol_train, y_notol_test = train_test_split(x_notol, y_notol, random_state=1)
    x_train, y_train = pd.concat([x_tol_train, x_notol_train], axis=0), pd.concat([y_tol_train, y_notol_train])
    x_test, y_test = pd.concat([x_tol_test, x_notol_test], axis=0), pd.concat([y_tol_test, y_notol_test])
    # サイズチェック
    print('shape of x and y', x.shape, y.shape)
    print('shape of train and test', x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    class_names = ['<=50K', '>50K']

    # 決定木
    # # グリッドサーチ
    classweight = [None, 'balanced']
    params = {'random_state': [1],
              'min_samples_leaf': [500],  # 学習データの3/4のため
              'min_impurity_decrease': [0, 0.1, 0.01, 0.001, 0.0001, 0.00001],
              'class_weight': classweight
              }
    clf = GridSearchCV(DecisionTreeClassifier(), params, cv=4, n_jobs=1, scoring='roc_auc')
    clf.fit(x_train, np.ravel(y_train))
    df_result = pd.DataFrame(clf.cv_results_)
    df_result. \
        to_csv('../output/gridsearch_result_dt1.csv')

    # # AUCが最良のパラメータで再学習
    min_impurity_decrease = clf.best_params_['min_impurity_decrease']
    class_weight = clf.best_params_['class_weight']
    dummy = DummyClassifier(strategy='most_frequent').fit(x_train, y_train)
    clf = DecisionTreeClassifier(random_state=1, min_samples_leaf=33,
                                 min_impurity_decrease=min_impurity_decrease, class_weight=class_weight)
    clf.fit(x_train, np.ravel(y_train))
    # # # 精度を評価する
    evaluate_model(clf, x_train, y_train, x_test, y_test, class_names)
    # # # 木のPDF出力する
    output_tree(clf, x_train, class_names)
    # # # 条件分岐を出力する
    output_branch(clf, class_names, x_train, y_train, x_test, y_test)


def evaluate_model(model, x_train, y_train, x_test, y_test, class_names):

    predict_train, predict_test = model.predict(x_train), model.predict(x_test)
    probability_train, probability_test = model.predict_proba(x_train), model.predict_proba(x_test)
    path_train, path_test = model.apply(x_train), model.apply(x_test)

    # 予測結果、予測確率（決定木の場合は条件番号も）を出力する
    predict_path_probability_train = np.c_[predict_train, probability_train[:, 0], probability_train[:, 1], path_train]
    predict_path_probability_test = np.c_[predict_test, probability_test[:, 0], probability_test[:, 1], path_test]
    df_train = pd.DataFrame(predict_path_probability_train, index=y_train.index,
                            columns=['train_pred', 'train_probability({})'.format(class_names[0]),
                                     'train_probability({})'.format(class_names[1]), 'train_path'])
    df_test = pd.DataFrame(predict_path_probability_test, index=y_test.index,
                           columns=['test_pred', 'test_probability({})'.format(class_names[0]),
                                    'test_probability({})'.format(class_names[1]), 'test_path'])

    df_train['y'], df_test['y'] = y_train, y_test

    paths_train = df_train['train_path'].drop_duplicates().astype(int).sort_values().reset_index(drop=True)
    paths_test = df_test['test_path'].drop_duplicates().astype(int).sort_values().reset_index(drop=True)

    paths_jp_train = pd.Series(['PATH{0}'.format(i + 1) for i in range(len(paths_train))])
    paths_jp_test = pd.Series(['PATH{0}'.format(i + 1) for i in range(len(paths_test))])

    df_paths_train = pd.DataFrame(pd.concat([paths_train, paths_jp_train], axis=1))
    df_paths_test = pd.DataFrame(pd.concat([paths_test, paths_jp_test], axis=1))

    df_paths_train.columns = ['train_path', 'train_path_no']
    df_paths_test.columns = ['test_path', 'test_path_no']

    df_res_train = pd.merge(df_train, df_paths_train)
    df_res_test = pd.merge(df_test, df_paths_test)
    df_res_train.index, df_res_test.index = df_train.index, df_test.index

    df_res_train.to_csv('../output/df_res_train.csv', encoding='utf-16')
    df_res_test.to_csv('../output/df_res_test.csv', encoding='utf-16')


def output_tree(dt, x, class_names):
    dot_data = tree.export_graphviz(dt,  # 決定木オブジェクトを一つ指定する
                                    out_file=None,  # ファイルは介さずにGraphvizにdot言語データを渡すのでNone
                                    filled=True,  # Trueにすると、分岐の際にどちらのノードに多く分類されたのか色で示してくれる
                                    rounded=True,  # Trueにすると、ノードの角を丸く描画する。
                                    feature_names=x.columns,  # これを指定しないとチャート上で特徴量の名前が表示されない
                                    # class_names=class_names,  # これを指定しないとチャート上で分類名が表示されない
                                    special_characters=True  # 特殊文字を扱えるようにする
                                    )
    graph = pydotplus.graphviz.graph_from_dot_data(dot_data)
    graph.write_png('../output/tree.png')


def output_branch(dt, class_names, x_train, y_train, x_test, y_test):
    # 分岐条件の変数名（IDと名称の対応は学習データの順番？）
    # -2は分岐条件がない（リーフである）ことを示す
    branch_name = [x_train.columns[i] if i != -2 else '' for i in dt.tree_.feature]

    # 到達ノードと予測クラスを取得する
    # # 学習データ
    path_train = dt.apply(x_train)
    pred_train = dt.predict(x_train)
    # # 評価データ
    path_test = dt.apply(x_test)
    pred_test = dt.predict(x_test)

    # 分岐条件を二次元で取得する
    result, result_yn, last_node = get_result_dt(dt)

    result_jp = []
    result_last_node = []
    all_impurity = 1 - ((y_train == 0).sum() / len(np.ravel(y_train))) ** 2\
                     - ((y_train == 1).sum() / len(np.ravel(y_train))) ** 2

    for i in range(result.shape[0]):
        row_branch, row_column, row_threshold, row_yn, row_impurity, row_samples, row_value, row_class, row_last_node = \
            [], [], [], [], [], [], [], [], []

        for j, k in zip(result.iloc[i, :].astype(int), range(result_yn.shape[1])):
            if dt.tree_.threshold[j] != -2.0:
                row_branch.append('{0} > {1} {2}'.format(branch_name[j], dt.tree_.threshold[j], result_yn.iloc[i, k]))
                row_column.append(branch_name[j])
                row_threshold.append(dt.tree_.threshold[j])
                row_yn.append(result_yn.iloc[i, k])
            else:
                row_branch.append('')
                row_column.append('')
                row_threshold.append('')
                row_yn.append('')

        result_jp.append(row_branch + row_column + row_threshold + row_yn)

        # 予測クラス
        row_last_node.append(class_names[np.argmax(dt.tree_.value[last_node[i]])])
        # 全体の不純度
        row_last_node.append(all_impurity)
        # 学習データ
        # # 最終ノードの到達数
        row_last_node.append((path_train == last_node[i]).astype(int).sum())
        # # 最終ノードのクラス別到達数
        row_last_node.append(((path_train == last_node[i]) & (pred_train == 0)).astype(int).sum())
        row_last_node.append(((path_train == last_node[i]) & (pred_train == 1)).astype(int).sum())
        # # 実際のクラス別数
        row_last_node.append(((path_train == last_node[i]) & (y_train == 0)).astype(int).sum())
        row_last_node.append(((path_train == last_node[i]) & (y_train == 1)).astype(int).sum())
        # # 不純度
        row_last_node.append(1 - (((path_train == last_node[i]) & (y_train == 0)).astype(int).sum() /
                                  (path_train == last_node[i]).astype(int).sum()) ** 2
                               - (((path_train == last_node[i]) & (y_train == 1)).astype(int).sum() /
                                  (path_train == last_node[i]).astype(int).sum()) ** 2)
        # 評価データ
        # # 最終ノードの到達数
        row_last_node.append((path_test == last_node[i]).astype(int).sum())
        # # 最終ノードのクラス別到達数
        row_last_node.append(((path_test == last_node[i]) & (pred_test == 0)).astype(int).sum())
        row_last_node.append(((path_test == last_node[i]) & (pred_test == 1)).astype(int).sum())
        # # 実際のクラス別数
        row_last_node.append(((path_test == last_node[i]) & (y_test == 0)).astype(int).sum())
        row_last_node.append(((path_test == last_node[i]) & (y_test == 1)).astype(int).sum())
        # # 不純度
        row_last_node.append(1 - (((path_test == last_node[i]) & (y_test == 0)).astype(int).sum() /
                                  (path_test == last_node[i]).astype(int).sum()) ** 2
                               - (((path_test == last_node[i]) & (y_test == 1)).astype(int).sum() /
                                  (path_test == last_node[i]).astype(int).sum()) ** 2)

        result_last_node.append(row_last_node)

    df_result_jp = pd.DataFrame(result_jp)
    df_result_last_node = pd.DataFrame(result_last_node)

    df_result = pd.concat([df_result_jp, df_result_last_node], axis=1)
    df_result.index = ['PATH{0}'.format(i) for i in range(1, df_result_jp.shape[0] + 1)]

    columns_branch = ['depth{0}_branch'.format(i) for i in range(1, int(df_result_jp.shape[1] / 4) + 1)]
    columns_column = ['depth{0}_column'.format(i) for i in range(1, int(df_result_jp.shape[1] / 4) + 1)]
    columns_threshold = ['depth{0}_threshold'.format(i) for i in range(1, int(df_result_jp.shape[1] / 4) + 1)]
    columns_yn = ['depth{0}_yn'.format(i) for i in range(1, int(df_result_jp.shape[1] / 4) + 1)]
    columns2 = ['pred_class', 'all_impurity', 'train_samples', 'train_pred_non_tol', 'train_pred_tol',
                'train_real_non_tol', 'train_real_tol', 'train_impurity',
                'test_samples', 'test_pred_non_tol', 'test_pred_tol',
                'test_real_non_tol', 'test_real_tol', 'test_impurity']
    df_result.columns = columns_branch + columns_column + columns_threshold + columns_yn + columns2

    # 適合率のカラムを作成する
    # # 学習データ
    df_tmp = df_result[['pred_class', 'train_samples', 'train_real_non_tol', 'train_real_tol']]
    df_tmp.columns = ['pred_class', 'samples', 'real_non_tol', 'real_tol']
    # # 評価データ
    df_tmp = df_result[['pred_class', 'test_samples', 'test_real_non_tol', 'test_real_tol']]
    df_tmp.columns = ['pred_class', 'samples', 'real_non_tol', 'real_tol']

    df_result.to_csv('../output/result_dt.csv', encoding='utf-16')

    return df_result


def get_result_dt(dt):
    """
    決定木の条件分岐を扱いやすい形式に変換する
    :param dt: 構築後の決定木モデル
    :return:
      ■パスごとに、上から下までのノード番号
      ■パスごとに、上から下までの閾値
      ■最終ノードの番号
    """
    left = dt.tree_.children_left
    right = dt.tree_.children_right
    get_left = []
    result_row, result_row_yn = [], []
    result = []  # どのノードとどのノードが結合しているかテーブルに格納用
    result_yn = []  # 各ノードのYESNO格納用
    last_node = []  # 各条件の最終ノード格納用

    i, j = 0, 0  # 親、子供

    result_row.append(i)  # 項目nにiを登録
    j = left[i]  # jに左の子[i]を登録

    n = 1  # 登録する深さ（0は一番上）

    ''' 1レコード目 '''
    # j（左の子）が存在する場合
    while j != -1:

        get_left.append(i)  # 左側取得リストに親を登録
        result_row.append(j)  # 項目nに子を登録
        result_row_yn.append('no')

        # 親と子を1深くする
        i, j, n = one_depth(i, j, n, left)

    # 結果格納用変数を作成する
    result.append(copy.deepcopy(result_row))
    result_yn.append(copy.deepcopy(result_row_yn))
    last_node.append(i)

    ''' 2レコード目以降 '''
    # 左側取得リストの中身がある場合
    while len(get_left) > 0:

        i = get_left.pop(-1)  # 左側取得リストの最後の要素を親とする（リストからは削除）
        n = int(np.where(pd.Series(result_row) == i)[0]) + 1  # 親の深さ+1を取得
        j = right[i]  # 右の子を取得
        result_row[n] = j  # 項目nにjを登録
        result_row_yn[n-1] = 'yes'

        # 親と子を1深くする
        i, j, n = one_depth(i, j, n, left)

        # j（左の子）が存在する場合
        while j != -1:
            get_left.append(i)  # 左側取得リストに親を登録

            if n >= len(result_row):
                result_row.append(j)
                result_row_yn.append('no')
            else:
                result_row[n] = j  # 項目nに子を登録
                result_row_yn[n-1] = 'no'

            # 親と子を1深くする
            i, j, n = one_depth(i, j, n, left)

        # 結果用変数に登録する
        result_row[n:] = np.full(len(result_row[n:]), -1)  # 以降の項目を-1に更新
        result_row_yn[n-1:] = np.full(len(result_row_yn[n-1:]), '')
        result.append(copy.deepcopy(result_row))
        result_yn.append(copy.deepcopy(result_row_yn))

        last_node.append(i)

    max_depth = pd.Series([len(result_row) for result_row in result]).max()

    result2 = []
    result_yn2 = []
    i = 0
    for _ in result:
        result_row, result_row_yn = result[i], result_yn[i]
        nn = max_depth - len(result_row)

        result_row[len(result_row):], result_row_yn[len(result_row_yn):] = np.full(nn, -1), np.full(nn, -1)
        result2.append(copy.deepcopy(result_row))
        result_yn2.append(copy.deepcopy(result_row_yn))
        i += 1

    # 結果用変数を二次元にする
    result = pd.DataFrame(result2)
    result_yn = pd.DataFrame(result_yn2)

    return result, result_yn, last_node


def one_depth(i, j, n, left):
    # 親と（左の）子を1深くし、nをインクリメントする
    return j, left[j], n + 1


if __name__ == "__main__":

    print(datetime.today(), 'START')

    main()

    print(datetime.today(), 'END')





