from bio_embeddings.embed import ProtTransBertBFDEmbedder
import sys
import numpy as np
import csv
import os

curDir = os.path.dirname(os.path.realpath(__file__)) + '/'
def create_csv(path):
    with open(path, "w+", newline='') as file:
        csv_file = csv.writer(file)
        head = ["sequences", "features"]
        csv_file.writerow(head)

def append_csv(path, datas):
    with open(path, "a+", newline='') as file:  # 处理csv读写时不同换行符  linux:\n    windows:\r\n    mac:\r
        csv_file = csv.writer(file)
        # datas = [["hoojjack", "boy"], ["hoojjack1", "boy"]]
        csv_file.writerows(datas)
def embedding(X_test):
    embedder = ProtTransBertBFDEmbedder()
    path = curDir + 'embedding.csv'
    create_csv(path)
    for i in range(X_test.shape[0]):
        embedding = embedder.embed(X_test[i])
        np.set_printoptions(threshold=sys.maxsize)  # 全部输出
        print(embedding,(np.array(embedding)).shape)