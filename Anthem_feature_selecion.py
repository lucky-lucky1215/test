# encoding:utf-8
import glob
import os
import sys

import umap
from scipy import stats
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.ensemble import BaggingClassifier
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, matthews_corrcoef, roc_auc_score, average_precision_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import pandas as pd
import numpy as np


def Ttest(X, Y, ttest_num=0):
    Y1 = np.where(Y == 1)[0]
    Y0 = np.where(Y == 0)[0]
    p_list = []
    for i in range(len(X[0])):
        p_l = stats.levene(X[Y1, i], X[Y0, i])[1]
        equal_var = [True if p_l > 0.05 else False]
        p_list.append(stats.ttest_ind(X[Y1, i], X[Y0, i], equal_var=equal_var)[1])

    return p_list, None


def Wtest(X, Y, wtest_num=0):
    Y1 = np.where(Y == 1)[0]
    Y0 = np.where(Y == 0)[0]
    p_list = []
    for i in range(len(X[0])):
        p_list.append(stats.ranksums(X[Y1, i], X[Y0, i])[1])
    return p_list, None


# def Chi2(X, Y, ctest_num=0):
#     p_list = chi2(X, Y)[1]
#     return p_list, None


def RF(X, Y, rtest_num=0):
    forest = RandomForestClassifier(random_state=0, n_jobs=1)
    forest.fit(X, Y)
    importance = forest.feature_importances_
    return 1 / (importance + 1e-10), None


def LR_RFE(X, Y, lrtest_num=0):
    clf = LinearRegression()
    rfe = RFE(clf, n_features_to_select=1)
    rfe.fit(X, Y)
    rank = rfe.ranking_
    return rank, None


def SVM_RFE(X, Y, srtest_num=0):
    clf = SVC(kernel='linear', random_state=0)
    rfe = RFE(clf, n_features_to_select=1)
    rfe.fit(X, Y)
    rank = rfe.ranking_
    return rank, None


def init_clf():
    '''
    init classification
    :return: classifiers and their parameters
    '''
    clfs = {'svm': svm.SVC(probability=True), 'xgb': XGBClassifier(probability=True,use_label_encoder=False),
            'knn': KNeighborsClassifier(), 'nb': GaussianNB(),
            'bagging': BaggingClassifier(), 'lr': LogisticRegression(),
            'dtree': DecisionTreeClassifier()}
    return clfs


def visual(data, label, x=0, y=1):
    print(data.min(), ',', data.max(), ',', data.mean(), ',', data.std())
    import matplotlib.pyplot as plt
    pos = data[np.where(label == 1)[0]][:, :6]  # (n,2)
    print(len(pos))
    neg = data[np.where(label == 0)[0]][:, :6]
    print(len(neg))
    plt.scatter(pos[:, x], pos[:, y], c='r', marker='x')
    plt.scatter(neg[:, x], neg[:, y], c='b', marker='.')
    plt.show()


def fs(train, val, test , appendix,pep_length):
    """

    [train_feature, train_label], [eval_feature, eval_label], [test_feature, test_label]
    :param train: [train_feature, train_label]
    :param val: [eval_feature, eval_label]
    :param test:  [test_feature, test_label]
    :return:
    """
    # 注意，这里经pipeline进行特征处理、SVC模型训练之后，得到的直接就是训练好的分类器clf
    # ('SKB', SelectKBest(k=2000)),
    results = []
    clfs = init_clf()
    best_acc = {}
    for clf_name in clfs.keys():
        best_acc[clf_name] = 0

    sd = StandardScaler()
    train[0] = sd.fit_transform(train[0], train[1])
    val[0] = sd.transform(val[0])
    test[0] = sd.transform(test[0])

    for fun in [Ttest, Wtest, RF, SVM_RFE, LR_RFE]:
        # for j in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
        #     pca = PCA(n_components=j)
        #     pca.fit(train[0], train[1])
        #     X_train = pca.transform(train[0])
        #     X_val = pca.transform(val[0])
        for j in [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]:
            select = umap.UMAP(n_components=j)
            select.fit(train[0], train[1])
            X_train = select.transform(train[0])
            X_val = select.transform(val[0])
            X_test = select.transform(test[0])
            for i in [0.55, 0.75, 0.95]:
                skb = SelectKBest(score_func=fun, k=int(X_train.shape[1]))
                skb.fit(X_train, train[1])
                skb.k = int(X_train.shape[1] * i)
                print('%.2f' % i + " " + fun.__name__)
                X_train_i = skb.transform(X_train)
                X_val_i = skb.transform(X_val)
                X_test_i = skb.transform(X_test)
                for clf_name in clfs.keys():
                    clf = clfs[clf_name]
                    clf.fit(X_train_i, train[1])
                    scorce = clf.score(X_val_i, val[1])
                    if scorce > best_acc[clf_name]:
                        best_acc[clf_name] = scorce
                        print("val best acc in %s is ：" % clf_name, best_acc)
                        y_pres = clf.predict(X_test_i)
                        y_probs = clf.predict_proba(X_test_i)
                        result = metrics(preds=y_pres, probs=y_probs[:, 1], labels=test[1])
                        result['clf'] = clf_name
                        result['val score'] = scorce
                        result['index num'] = '%.2f' % i
                        result['n_com'] = '%.2f' % j
                        result['fs'] = fun.__name__
                        results.append(pd.DataFrame(result, index=[0]))
                        print("test best acc in %s is ：" % clf_name, result)
    results = pd.concat(results)
    results.to_excel(os.path.join('%s_fe_%s.xlsx' % (pep_length, appendix)))
    return results

def reshape(a):
    a[0].shape = (len(a[1]), -1)
    a[1].shape = (-1,)

def metrics(preds, labels, probs):
    print(type(preds))
    print(preds.shape)
    print(type(labels))
    print(labels)
    acc = (preds == labels).mean()
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    sn = tp / (tp + fn)
    sp = tn / (tn + fp)
    mcc = matthews_corrcoef(labels, preds)
    auc = roc_auc_score(labels, probs)
    aupr = average_precision_score(labels, probs)
    return {
        "acc": acc,
        "mcc": mcc,
        "auc": auc,
        "aupr": aupr,
        "sn": sn,
        "sp": sp,
    }


if __name__ == '__main__':
    import pandas as pd

    dict = {"8": ["HLA-A_01_01"],
    # "HLA-A_01_01","HLA-A_02_01", "HLA-A_02_02", "HLA-A_02_03", "HLA-A_02_04", "HLA-A_02_05", "HLA-A_02_06",
    #            "HLA-A_02_07", "HLA-A_02_11", "HLA-A_02_12", "HLA-A_02_16", "HLA-A_02_17", "HLA-A_02_19", "HLA-A_02_20",
    #            "HLA-A_02_50", "HLA-A_03_01", "HLA-A_11_01", "HLA-A_23_01", "HLA-A_24_02", "HLA-A_24_03", "HLA-A_24_06",
    #            "HLA-A_24_13", "HLA-A_25_01", "HLA-A_26_01", "HLA-A_26_02", "HLA-A_26_03", "HLA-A_29_02", "HLA-A_30_01",
    #            "HLA-A_30_02", "HLA-A_31_01", "HLA-A_32_01", "HLA-A_32_07", "HLA-A_32_15", "HLA-A_33_01", "HLA-A_66_01",
    #            "HLA-A_68_01", "HLA-A_68_02", "HLA-A_68_23", "HLA-A_69_01", "HLA-A_80_01","HLA-B_07_02", "HLA-B_08_01",
    #            "HLA-B_13_02", "HLA-B_14_01", "HLA-B_14_02", "HLA-B_15_01", "HLA-B_15_02", "HLA-B_15_03", "HLA-B_15_09",
    #         "HLA-B_15_11", "HLA-B_15_17", "HLA-B_15_18", "HLA-B_15_42", "HLA-B_18_01", "HLA-B_18_03", "HLA-B_27_01","HLA-B_27_02",
    #         "HLA-B_27_03", "HLA-B_27_04", "HLA-B_27_05", "HLA-B_27_06", "HLA-B_27_07", "HLA-B_27_08",
    #        "HLA-B_27_09", "HLA-B_27_20", "HLA-B_35_01", "HLA-B_35_03", "HLA-B_35_08", "HLA-B_37_01", "HLA-B_38_01",
    #        "HLA-B_39_01","HLA-B_39_06", "HLA-B_39_24","HLA-B_40_01", "HLA-B_40_02", "HLA-B_41_01", "HLA-B_44_02",
            #            "HLA-B_44_03", "HLA-B_45_01", "HLA-B_45_06", "HLA-B_46_01", "HLA-B_48_01", "HLA-B_49_01", "HLA-B_50_01",
            #            "HLA-B_51_01", "HLA-B_51_08", "HLA-B_52_01", "HLA-B_53_01", "HLA-B_54_01", "HLA-B_56_01", "HLA-B_57_01",
            #            "HLA-B_57_03", "HLA-B_58_01", "HLA-B_73_01", "HLA-B_83_01","HLA-C_01_02", "HLA-C_02_02", "HLA-C_03_03","HLA-C_03_04", "HLA-C_04_01", "HLA-C_05_01",
           "9": ["HLA-C_06_02", "HLA-C_07_01", "HLA-C_07_02", "HLA-C_07_04",
           "HLA-C_08_02", "HLA-C_12_03", "HLA-C_14_02", "HLA-C_15_02", "HLA-C_16_01", "HLA-C_17_01"],
     "10": ["HLA-A_01_01", "HLA-A_02_01", "HLA-A_02_02", "HLA-A_02_03", "HLA-A_02_04", "HLA-A_02_05", "HLA-A_02_06",
            "HLA-A_02_07", "HLA-A_02_17", "HLA-A_03_01", "HLA-A_11_01", "HLA-A_23_01", "HLA-A_24_02", "HLA-A_24_06",
            "HLA-A_26_01", "HLA-A_29_02", "HLA-A_30_01", "HLA-A_30_02", "HLA-A_31_01", "HLA-A_32_01", "HLA-A_33_01",
            "HLA-A_68_01", "HLA-A_68_02", "HLA-A_69_01", "HLA-B_07_02", "HLA-B_08_01", "HLA-B_13_02", "HLA-B_14_02",
            "HLA-B_15_01", "HLA-B_18_01", "HLA-B_27_01", "HLA-B_27_02", "HLA-B_27_03", "HLA-B_27_04", "HLA-B_27_05",
            "HLA-B_27_06", "HLA-B_27_07", "HLA-B_27_08", "HLA-B_27_09", "HLA-B_35_01", "HLA-B_35_03", "HLA-B_35_08",
            "HLA-B_37_01", "HLA-B_39_01", "HLA-B_40_01", "HLA-B_40_02", "HLA-B_41_01", "HLA-B_44_02", "HLA-B_44_03",
            "HLA-B_44_27", "HLA-B_45_01", "HLA-B_46_01", "HLA-B_49_01", "HLA-B_50_01", "HLA-B_51_01", "HLA-B_53_01",
            "HLA-B_54_01", "HLA-B_56_01", "HLA-B_57_01", "HLA-B_57_03", "HLA-B_58_01", "HLA-C_01_02", "HLA-C_02_02",
            "HLA-C_03_03", "HLA-C_03_04", "HLA-C_04_01", "HLA-C_05_01", "HLA-C_06_02", "HLA-C_07_01", "HLA-C_07_02",
            "HLA-C_07_04", "HLA-C_08_02", "HLA-C_14_02", "HLA-C_16_01"],
     "11": ["HLA-A_01_01", "HLA-A_02_01", "HLA-A_02_03", "HLA-A_02_04", "HLA-A_02_05", "HLA-A_02_07", "HLA-A_03_01",
            "HLA-A_11_01", "HLA-A_23_01", "HLA-A_24_02", "HLA-A_24_06", "HLA-A_29_02", "HLA-A_31_01", "HLA-A_32_01",
            "HLA-A_68_01", "HLA-A_68_02", "HLA-B_07_02", "HLA-B_08_01", "HLA-B_15_01", "HLA-B_27_01", "HLA-B_27_02",
            "HLA-B_27_03", "HLA-B_27_04", "HLA-B_27_05", "HLA-B_27_06", "HLA-B_27_07", "HLA-B_27_08", "HLA-B_27_09",
            "HLA-B_35_01", "HLA-B_35_03", "HLA-B_35_08", "HLA-B_37_01", "HLA-B_39_01", "HLA-B_40_01", "HLA-B_40_02",
            "HLA-B_44_02", "HLA-B_44_03", "HLA-B_45_01", "HLA-B_46_01", "HLA-B_49_01", "HLA-B_51_01", "HLA-B_54_01",
            "HLA-B_56_01", "HLA-B_57_01", "HLA-B_57_03", "HLA-B_58_01", "HLA-C_01_02", "HLA-C_02_02", "HLA-C_03_03",
            "HLA-C_03_04", "HLA-C_04_01", "HLA-C_05_01", "HLA-C_06_02", "HLA-C_07_01", "HLA-C_07_02", "HLA-C_08_02",
            "HLA-C_16_01"],
     "12": ["HLA-A_01_01", "HLA-A_02_01", "HLA-A_03_01", "HLA-A_11_01", "HLA-A_24_02", "HLA-A_29_02", "HLA-A_31_01",
            "HLA-A_68_01", "HLA-A_68_02", "HLA-B_07_02", "HLA-B_08_01", "HLA-B_15_01", "HLA-B_27_01", "HLA-B_27_02",
            "HLA-B_27_03", "HLA-B_27_05", "HLA-B_27_07", "HLA-B_27_08", "HLA-B_27_09", "HLA-B_35_01", "HLA-B_40_01",
            "HLA-B_40_02", "HLA-B_44_02", "HLA-B_44_03", "HLA-B_51_01", "HLA-B_57_01", "HLA-B_57_03", "HLA-B_58_01",
            "HLA-C_01_02", "HLA-C_04_01", "HLA-C_05_01", "HLA-C_06_02", "HLA-C_07_01"],
     "13": ["HLA-A_01_01", "HLA-A_02_01", "HLA-A_03_01", "HLA-A_11_01", "HLA-A_24_02", "HLA-A_29_02", "HLA-A_31_01",
            "HLA-A_68_02", "HLA-B_07_02", "HLA-B_15_01", "HLA-B_27_01", "HLA-B_27_02", "HLA-B_27_05", "HLA-B_27_08",
            "HLA-B_27_09", "HLA-B_35_01", "HLA-B_44_02", "HLA-B_51_01", "HLA-B_57_01", "HLA-B_57_03", "HLA-B_58_01",
            "HLA-C_04_01", "HLA-C_05_01", "HLA-C_06_02"],
     "14": ["HLA-A_01_01", "HLA-A_02_01", "HLA-A_24_02", "HLA-A_68_02", "HLA-B_07_02", "HLA-B_15_01", "HLA-B_27_05",
            "HLA-B_27_09", "HLA-B_35_01", "HLA-B_57_01", "HLA-C_04_01", "HLA-C_05_01", "HLA-C_06_02"]}
    for key in ["8"]:
        for hla in dict[key]:
            train = pd.read_hdf(os.path.join("Anthem_feature/"+str(key), hla + '_' + 'train_embed.h5') )
            valid = pd.read_hdf(os.path.join("Anthem_feature/"+str(key), hla + '_' + 'valid_embed.h5') )
            test = pd.read_hdf(os.path.join("Anthem_feature/"+str(key), hla + '_' + 'test_embed.h5') )  # 读取h5
            train_= [0, 0]
            val_= [0, 0]
            test_ = [0, 0]
            test_[0], test_[1] = test.values[:, 0:-1], test.values[:, -1]
            val_[0], val_[1] = valid.values[:, 0:-1], valid.values[:, -1]
            train_[0], train_[1] = train.values[:, 0:-1], train.values[:, -1]
            visual(train_[0],train_[1])
            visual(val_[0], val_[1])
            visual( test_[0], test_[1])
            print("******************数据获取完毕****************")
            appendix = hla
            result = fs(train_, val_, test_,appendix, key)
            # results = pd.concat(results)
