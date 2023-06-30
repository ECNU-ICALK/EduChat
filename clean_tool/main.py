from math import ceil
import pickle

import numpy
import torch
from sentence_transformers import SentenceTransformer
import os
import json
from tqdm import trange
import torch.nn.functional as F

res = []
def solve_init(data,neigh):
    
    therd = 0.7
    # therd = 0.54
    pa = [i for i in range(len(data))]
    X = []
    for i in data:
        X.append(i[0])
    X = F.normalize(torch.Tensor(X), dim=1, p=2).numpy()
    from sklearn.neighbors import BallTree,KDTree
    tree = BallTree(X, leaf_size=64)


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
            if random.randint(0,1)==0:
                a1,b1 = b1,a1
            pa[a1] = b1

    def qry(X,neigh):
        dist, ind = tree.query(X, k=neigh)
        dist = 1 - dist
        return (dist[0],ind[0])

    from joblib import Parallel, delayed
    
    from tqdm import tqdm  
    import numpy as np
    X = X.reshape(len(X),1,-1)
    array_list = X.tolist()
    X = np.array([np.array(sublist) for sublist in array_list])
    # print(X[0].shape)
    Y = Parallel(n_jobs=64)(delayed(qry)(i,neigh) for i in tqdm(X))
    # print(Y)
    for i in trange(len(data)):
        dist = Y[i][0]
        ind = Y[i][1]
        for j in range(dist.shape[0]):
            if dist[j]>=therd:
                un(i,ind[j])
    cnt = 0
    for i in range(len(data)):
        if i==fid(i):
           res.append(data[i])
           cnt += 1
    print(cnt,len(data))

def solve(file_path):
    res2 = []
    for i in range(len(data)):
        res2.append(data[i][1])
    with open(file_path.replace(".pt",".jsonl"),"w",encoding="utf-8") as f:
        for i in res2:
            f.write(json.dumps(i,ensure_ascii=False)+"\n")

import sys
LANG = sys.argv[1]
block_size = int(sys.argv[2])
neigh = 512
file_path = f"./opensource_data/MIX_{LANG}.pt"
data = pickle.load(open(file_path,"rb"))

while block_size>1:
    if block_size==2:
        block_size=1
    block = ceil(len(data)/block_size)
    print(block_size)
    block_size//=4
    for i in range(0,len(data),block):
        solve_init(data[i:i+block],256)
    data = res
    res = []

solve_init(data,512)
data = res
res = []
solve(file_path)



