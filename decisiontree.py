# -*- encoding:utf-8 -*-
import os
import pandas as pd
import numpy as np
import pydotplus
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn import tree
from sklearn.externals import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix, classification_report, auc, roc_curve
from sklearn.model_selection import train_test_split, GridSearchCV

# 決定木の深さ
depth = 8


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

    columns = x.columns
    class_names = ['<=50K', '>50K']

    # 決定木
    # # グリッドサーチ
    classweight = [None, 'balanced']
    params = {'random_state': [1],
              'min_samples_leaf': [25],  # 学習データの3/4のため
              'min_impurity_decrease': [0, 0.1, 0.01, 0.001, 0.0001, 0.00001],
              'max_depth': [depth],
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
    clf = DecisionTreeClassifier(random_state=1, min_samples_leaf=33, max_depth=depth,
                                 min_impurity_decrease=min_impurity_decrease, class_weight=class_weight)
    clf.fit(x_train, np.ravel(y_train))
    # # # 精度を評価する
    evaluate_dt, df_res_dt = evaluate_model(clf, 'dt2', x_train, y_train, x_test, y_test)
    # # # 木のPDF出力する
    output_tree(clf, x_train, class_names)
    # # # 条件分岐を出力する
    df_result = output_branch(clf, class_names, x_train, y_train, x_test, y_test)


def evaluate_model(model, model_name, x_train, y_train, x_test, y_test):

    pred_train = model.predict(x_train)
    proba_train = model.predict_proba(x_train)
    pred_test = model.predict(x_test)
    proba_test = model.predict_proba(x_test)
    path_test = model.apply(x_test) if model_name[:2] == 'dt' else None

    # ConfusionMatrixを出力する
    print(datetime.today(),
          'confusion matrix  upper left:both notol  upper right:real notol/pred tol  '
          'lower left:real tol/pred notol  lower right:both tol')
    con_mat_tr, con_mat_ts = confusion_matrix(y_train, pred_train), confusion_matrix(y_test, pred_test)
    cla_rep_tr = classification_report(y_train, pred_train, target_names=['notol', 'tol'], digits=5)
    cla_rep_ts = classification_report(y_test, pred_test, target_names=['notol', 'tol'], digits=5)
    print(con_mat_tr)
    print(con_mat_ts)
    print(cla_rep_tr)
    print(cla_rep_ts)

    # 学習データ、ダミーデータ、評価データのROC曲線とAUC
    fpr_tr, tpr_tr, thresholds_tr = roc_curve(y_train, proba_train[:, 1])
    fpr_ts, tpr_ts, thresholds_ts = roc_curve(y_test, proba_test[:, 1])
    roc_auc_tr = auc(fpr_tr, tpr_tr)
    roc_auc_ts = auc(fpr_ts, tpr_ts)
    print('AUC_train', round(roc_auc_tr, 4))
    print('AUC_test', round(roc_auc_ts, 4))

    # テキストに保存する
    with open('../output/evaluate_{0}.txt'.format(model_name), mode='w') as f:
        f.write('train_shape' + str(x_train.shape) + '\n')
        f.write('test_shape' + str(x_test.shape) + '\n')
        f.write(str(model.get_params(deep=False)) + '\n')
        f.write('Confusion Matrix' + '\n')
        f.write(str(con_mat_tr[0][0]) + ',' + str(con_mat_tr[0][1]) + '\n')
        f.write(str(con_mat_tr[1][0]) + ',' + str(con_mat_tr[1][1]) + '\n')
        f.write(str(con_mat_ts[0][0]) + ',' + str(con_mat_ts[0][1]) + '\n')
        f.write(str(con_mat_ts[1][0]) + ',' + str(con_mat_ts[1][1]) + '\n')
        f.write(str(cla_rep_tr) + '\n')
        f.write(str(cla_rep_ts) + '\n')
        f.write('AUC_train' + ',' + str(round(roc_auc_tr, 4)) + '\n')
        f.write('AUC_test' + ',' + str(round(roc_auc_ts, 4)))

    # ROC曲線を描く
    fig = plt.figure(figsize=(15, 10))
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.plot(fpr_tr, tpr_tr, color='b', label='train (area={0})'.format(round(roc_auc_tr, 3)))
    ax1.plot(fpr_ts, tpr_ts, color='r', label='test (area={0})'.format(round(roc_auc_ts, 3)))
    ax1.set_xlabel('False Positivie Rate')
    ax1.set_ylabel('True Positivie Rate')
    ax1.set_xlim(0.0, 1.0)
    ax1.set_ylim(0.0, 1.05)
    ax1.set_title('Receiver operating characteristic example')
    ax1.legend(loc='lower right')
    plt.savefig('../output/ROCcurve_{0}.png'.format(model_name))

    # 予測結果、予測確率（決定木の場合は条件番号も）を出力する
    if model_name[:2] == 'rf':
        pred_proba_test = np.c_[pred_test, proba_test[:, 1], proba_test[:, 0]]
        df_res = pd.DataFrame(pred_proba_test, index=y_test.index, columns=['pred', 'proba_tol', 'proba_notol'])
    else:
        pred_path_proba_test = np.c_[pred_test, proba_test[:, 1], proba_test[:, 0], path_test]
        df_res = pd.DataFrame(pred_path_proba_test, index=y_test.index,
                              columns=['pred', 'proba_tol', 'proba_notol', 'path'])
    df_res['y'] = y_test
    df_res['index'] = df_res.index
    if model_name[:2] == 'dt':
        paths = df_res['path'].drop_duplicates().astype(int).sort_values().reset_index(drop=True)
        paths_jp = pd.Series(['条件{0}'.format(i + 1) for i in range(len(paths))])
        df_paths = pd.DataFrame(pd.concat([paths, paths_jp], axis=1))
        df_paths.columns = ['path', 'path_jp']
        df_res = pd.merge(df_res, df_paths)
    df_res.to_csv('../output/df_res_{0}.csv'.format(model_name), encoding='utf-16')

    conmat_auc = [con_mat_ts[1, 1], con_mat_ts[1, 0], con_mat_ts[0, 1], con_mat_ts[0, 0],
                  round(roc_auc_tr, 4), round(roc_auc_ts, 4)]

    return conmat_auc, df_res


def output_importances(model, x, clf, select_num=None):
    # 特徴量の重要度を取得する
    feature_imps = clf.feature_importances_
    # 特徴量の名前
    label = x.columns[0:]
    # 必要な項目抽出用
    select_columns = []
    # 特徴量の重要度順（降順）
    indices = np.argsort(feature_imps)[::-1]

    with open(os.path.join('../output/importances_{0}.csv'.format(model)), mode='w') as f:
        for i in range(0, select_num):

            # 上位100の変数を出力する＆取り出す
            line = str(i + 1) + "," + str(label[indices[i]]) + "," + str(feature_imps[indices[i]]) \
                   + "," + str(label[indices[i]])
            # print(line)
            f.write(str(line) + '\n')

            select_columns.append(str(label[indices[i]]))

    return select_columns


def output_tree(dt, x, class_names):
    dot_data = tree.export_graphviz(dt,  # 決定木オブジェクトを一つ指定する
                                    out_file=None,  # ファイルは介さずにGraphvizにdot言語データを渡すのでNone
                                    filled=True,  # Trueにすると、分岐の際にどちらのノードに多く分類されたのか色で示してくれる
                                    rounded=True,  # Trueにすると、ノードの角を丸く描画する。
                                    feature_names=x.columns,  # これを指定しないとチャート上で特徴量の名前が表示されない
                                    class_names=class_names,  # これを指定しないとチャート上で分類名が表示されない
                                    special_characters=True  # 特殊文字を扱えるようにする
                                    )
    graph = pydotplus.graphviz.graph_from_dot_data(dot_data)
    graph.write_png('../output/tree.png')


def one_depth(i, j, n, left):
    # 親と（左の）子を1深くし、nをインクリメントする
    return j, left[j], n + 1


def output_branch(dt, class_names, x_train, y_train, x_test, y_test):
    # 分岐条件の変数名（IDと名称の対応は学習データの順番？）
    # -2は分岐条件がない（リーフである）ことを示す
    branch_name = [x_train.columns[i] if i != -2 else '' for i in dt.tree_.feature]

    # 到達ノードと予測クラスを取得する
    # # 学習データ
    path_train = dt.apply(x_train)
    pred_train = dt.predict(x_train)
    proba_train = dt.predict_proba(x_train)
    # # 評価データ
    path_test = dt.apply(x_test)
    pred_test = dt.predict(x_test)
    proba_test = dt.predict_proba(x_test)

    # 分岐条件を二次元で取得する
    result, result_yn, last_node = get_result_dt(dt)

    result_jp = []
    result_last_node = []
    all_impurity = 1 - ((y_train['y'] == 0).sum() / len(np.ravel(y_train))) ** 2\
                     - ((y_train['y'] == 1).sum() / len(np.ravel(y_train))) ** 2

    for i in range(result.shape[0]):
        row_branch, row_threshold, row_impurity, row_samples, row_value, row_class, row_last_node = \
            [], [], [], [], [], [], []

        for j, k in zip(result[i, :].astype(int), range(result_yn.shape[1])):
            if dt.tree_.threshold[j] != -2.0:
                row_branch.append('{0} {1}'.format(branch_name[j], result_yn[i, k]))
                row_threshold.append(dt.tree_.threshold[j])
            else:
                row_branch.append('')
                row_threshold.append('')

        result_jp.append(row_branch + row_threshold)

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
        row_last_node.append(((path_train == last_node[i]) & (y_train['y'] == 0)).astype(int).sum())
        row_last_node.append(((path_train == last_node[i]) & (y_train['y'] == 1)).astype(int).sum())
        # # 不純度
        row_last_node.append(1 - (((path_train == last_node[i]) & (y_train['y'] == 0)).astype(int).sum() /
                                  (path_train == last_node[i]).astype(int).sum()) ** 2
                               - (((path_train == last_node[i]) & (y_train['y'] == 1)).astype(int).sum() /
                                  (path_train == last_node[i]).astype(int).sum()) ** 2)
        # 評価データ
        # # 最終ノードの到達数
        row_last_node.append((path_test == last_node[i]).astype(int).sum())
        # # 最終ノードのクラス別到達数
        row_last_node.append(((path_test == last_node[i]) & (pred_test == 0)).astype(int).sum())
        row_last_node.append(((path_test == last_node[i]) & (pred_test == 1)).astype(int).sum())
        # # 実際のクラス別数
        row_last_node.append(((path_test == last_node[i]) & (y_test['y'] == 0)).astype(int).sum())
        row_last_node.append(((path_test == last_node[i]) & (y_test['y'] == 1)).astype(int).sum())
        # # 不純度
        row_last_node.append(1 - (((path_test == last_node[i]) & (y_test['y'] == 0)).astype(int).sum() /
                                  (path_test == last_node[i]).astype(int).sum()) ** 2
                               - (((path_test == last_node[i]) & (y_test['y'] == 1)).astype(int).sum() /
                                  (path_test == last_node[i]).astype(int).sum()) ** 2)

        result_last_node.append(row_last_node)

    df_result_jp = pd.DataFrame(result_jp)
    df_result_last_node = pd.DataFrame(result_last_node)

    df_result = pd.concat([df_result_jp, df_result_last_node], axis=1)
    df_result.index = ['条件{0}'.format(i) for i in range(1, df_result_jp.shape[0] + 1)]

    columns_branch = ['項目{0}_条件'.format(i) for i in range(1, int(df_result_jp.shape[1] / 2) + 1)]
    columns_threshold = ['項目{0}_閾値'.format(i) for i in range(1, int(df_result_jp.shape[1] / 2) + 1)]
    columns2 = ['pred_class', 'all_impurity', 'train_samples', 'train_pred_non_tol', 'train_pred_tol',
                'train_real_non_tol', 'train_real_tol', 'train_impurity',
                'test_samples', 'test_pred_non_tol', 'test_pred_tol',
                'test_real_non_tol', 'test_real_tol', 'test_impurity']
    df_result.columns = columns_branch + columns_threshold + columns2

    # 適合率のカラムを作成する
    # # 学習データ
    df_tmp = df_result[['pred_class', 'train_samples', 'train_real_non_tol', 'train_real_tol']]
    df_tmp.columns = ['pred_class', 'samples', 'real_non_tol', 'real_tol']
    # # 評価データ
    df_tmp = df_result[['pred_class', 'test_samples', 'test_real_non_tol', 'test_real_tol']]
    df_tmp.columns = ['pred_class', 'samples', 'real_non_tol', 'real_tol']

    df_result.to_csv('../output/result_dt_{0}.csv', encoding='utf-16')

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
    max_depth = depth + 1  # 出力する深さ
    # result_row = np.full(int(math.log2(max(right))) + 1, -1)
    # result_row_yn = np.full(int(math.log2(max(right))) + 1, '   ')
    result_row = np.full(max_depth, -1)
    result_row_yn = np.full(max_depth, '   ')
    result = []  # どのノードとどのノードが結合しているかテーブルに格納用
    result_yn = []  # 各ノードのYESNO格納用
    last_node = []  # 各条件の最終ノード格納用

    i, j = 0, 0  # 親、子供

    result_row[0] = i  # 項目nにiを登録
    j = left[i]  # jに左の子[i]を登録

    n = 1  # 登録する深さ（0は一番上）

    ''' 1レコード目 '''
    # j（左の子）が存在する場合
    while j != -1:

        get_left.append(i)  # 左側取得リストに親を登録
        result_row[n] = j  # 項目nに子を登録
        result_row_yn[n-1] = 'no'

        # 親と子を1深くする
        i, j, n = one_depth(i, j, n, left)

    # 結果格納用変数を作成する
    result = np.append(result, result_row)
    result_yn = np.append(result_yn, result_row_yn)
    last_node.append(i)

    ''' 2レコード目以降 '''
    # 左側取得リストの中身がある場合
    while len(get_left) > 0:

        i = get_left.pop(-1)  # 左側取得リストの最後の要素を親とする（リストからは削除）
        n = int(np.where(result_row == i)[0]) + 1  # 親の深さ+1を取得
        j = right[i]  # 右の子を取得
        result_row[n] = j  # 項目nにjを登録
        result_row_yn[n-1] = 'yes'

        # 親と子を1深くする
        i, j, n = one_depth(i, j, n, left)

        # j（左の子）が存在する場合
        while j != -1:
            get_left.append(i)  # 左側取得リストに親を登録
            result_row[n] = j  # 項目nに子を登録
            result_row_yn[n-1] = 'no'

            # 親と子を1深くする
            i, j, n = one_depth(i, j, n, left)

        # 結果用変数に登録する
        result_row[n:] = np.full(len(result_row[n:]), -1)  # 以降の項目を-1に更新
        result_row_yn[n-1:] = np.full(len(result_row_yn[n-1:]), '')
        result = np.append(result, result_row)
        result_yn = np.append(result_yn, result_row_yn)
        last_node.append(i)

    # 結果用変数を二次元にする
    result = result.reshape(-1, max_depth)
    result_yn = result_yn.reshape(-1, max_depth)

    return result, result_yn, last_node


if __name__ == "__main__":

    print(datetime.today(), 'START')

    main()

    print(datetime.today(), 'END')





