import sys, time, os, io, csv, math, datetime, csv, re
import numpy as np
import pandas as pd
# from bio_embed import *
from sklearn.model_selection import train_test_split

starttime = time.time()

curDir = os.path.dirname(os.path.realpath(__file__)) + '/'
HLA_seq = pd.read_csv(curDir + 'MHC_pseudo.dat', sep='\t')
seqs = {}
for i in range(len(HLA_seq)):
    seqs[HLA_seq.HLA[i]] = HLA_seq.sequence[i]


def transform(HLA, peptide):
    data = HLA + peptide
    seq = data + 'X' * (49 - len(data))
    return seq


def read_and_prepare(file):
    data = pd.read_csv(file)
    data['cost_cents'] = data.apply(
        lambda row: transform(
            HLA=seqs[row['HLA']],
            peptide=row['peptide']),
        axis=1)
    # print(data)
    return np.vstack(data.cost_cents)


file = curDir + 'train-data.csv'
fname = file.split('/')[-1]
fname1 = fname.split('.')[0]
df = pd.read_csv(file)
X_test = read_and_prepare(file)
# print(X_test)
# embedding(X_test)

endtime = time.time()
dtime = endtime - starttime

print("程序运行时间：%.8s s" % dtime)  # 显示到微秒

#
# print(df['Label'].value_counts())
# X = df.drop('Label',1)
# y = df['Label']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1,stratify=y)  #X是dataframe，y是Series

def train_test_val_split(df,ratio_train,ratio_test,ratio_val):
    train, middle = train_test_split(df,test_size=1-ratio_train,random_state=1,stratify=df['Label'])
    ratio=ratio_val/(1-ratio_train)
    test,validation =train_test_split(middle,test_size=ratio,random_state=1,stratify=middle['Label'])
    return train,test,validation


train,test,validation = train_test_val_split(df, 0.7, 0.1, 0.2)
print(train['Label'].value_counts(),test['Label'].value_counts(),validation['Label'].value_counts())
