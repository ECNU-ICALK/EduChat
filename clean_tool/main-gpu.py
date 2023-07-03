from math import ceil
import pickle
import sys
import numpy
import torch
# from sentence_transformers import SentenceTransformer
import os
import json
from tqdm import trange
import torch.nn.functional as F
import pandas as pd
sys.setrecursionlimit(100000000)

def solve_init(data, neigh, id):
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(id)
    res = []
    therd = 0.65
    pa = [i for i in range(len(data))]
    X = []
    for i in data:
        X.append(i[0])
    X = F.normalize(torch.Tensor(X), dim=1, p=2).numpy()
    import cudf

    # 将numpy数组转换为cudf dataframe

    # from sklearn.neighbors import BallTree,KDTree
    from cuml.neighbors import NearestNeighbors as cuNearestNeighbors
    knn_cuml = cuNearestNeighbors()
    knn_cuml.fit(X)

    # tree = NearestNeighbors(X, leaf_size=64)
    # tree = BallTree(X, leaf_size=64)

    def fid(x):
        if x == pa[x]:
            return x
        pa[x] = fid(pa[x])
        return pa[x]

    def un(x, y):
        a1 = fid(x)
        b1 = fid(y)
        if a1 != b1:
            import random
            if random.randint(0, 1) == 0:
                a1, b1 = b1, a1
            pa[a1] = b1

    def qry(X, neigh):
        dist, ind = knn_cuml.kneighbors(X, neigh)
        dist = 1 - dist
        return (dist[0], ind[0])

    from joblib import Parallel, delayed

    from tqdm import tqdm
    import numpy as np
    X = X.reshape(len(X), 1, -1)
    array_list = X.tolist()
    X = np.array([np.array(sublist) for sublist in array_list])
    Y = []
    for i in tqdm(X):
        Y.append(qry(i, neigh))

    # print(Y)
    for i in trange(len(data)):
        dist = Y[i][0]
        ind = Y[i][1]
        for j in range(dist.shape[0]):
            if dist[j] >= therd:
                un(i, ind[j])
    cnt = 0
    for i in range(len(data)):
        if i == fid(i):
            res.append(data[i][2])
            cnt += 1
    print(cnt, len(data))
    return res


def solve(file_path):
    res2 = []
    for i in range(len(data)):
        res2.append(data[i][1])
    with open(file_path.replace(".pt", ".jsonl"), "w", encoding="utf-8") as f:
        for i in res2:
            f.write(json.dumps(i, ensure_ascii=False) + "\n")


import sys

LANG = sys.argv[1]
block_size = int(sys.argv[2])
neigh = 512
file_path = f"./data/MIX_{LANG}.pt"
data = pickle.load(open(file_path, "rb"))
for i in range(len(data)):
    data[i] = [data[i][0],data[i][1],i]

while block_size >= 1:
    if block_size == 2:
        block_size=1
    block = ceil(len(data) / block_size)
    print(block_size)
    block_size //= 4
    from joblib import Parallel, delayed
    if block_size!=0:
        res = Parallel(n_jobs=8)(delayed(solve_init)(data[i:i + block], 256, i//block % 8) for i in range(0, len(data), block))
    else:
        res = [solve_init(data,512,0)]
    l = 0
    tmp = []
    for j in res:
        for k in j:
            tmp.append(data[k])
    data = tmp
    for i in range(len(data)):
        data[i][-1]=i
    res = []

solve(file_path)



