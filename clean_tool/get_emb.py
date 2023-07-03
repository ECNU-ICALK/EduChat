from sentence_transformers import SentenceTransformer
import os
import json
import torch
sentences = ["This is an example sentence", "Each sentence is converted"]
import sys  # 导入sys模块
sys.setrecursionlimit(1000000) 
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
model.eval()
res = []
mix_data = []
def solve(model,path):
    with open(path,"r",encoding="utf-8") as f:
        data = [eval(i) for i in f.readlines()]
    a = []
    b = []
    for i in range(len(data)):
        if "INSTRUCTION" in data[i]:
            nd = "User: "+data[i]["INSTRUCTION"]+"\n Assistant:"+data[i]["RESPONSE"]
        else:
            if len(data[i]["thread"]["text"])==0:
                continue
            nd = "User: "+data[i]["thread"]["text"]+"\n Assistant:"+data[i]["thread"]["replies"][0]["text"]
        a.append(nd)
        b.append(data[i])
        mix_data.append(data[i])


    interval = 32
    from tqdm import trange
    for i in trange(0,len(a),interval):
        embeddings = model.encode(a[i:i+interval])
        for j in range(len(embeddings)):
            res.append((embeddings[j],b[i+j]))



import sys
LANG = sys.argv[1]


for root, dirs, files in os.walk(f'./data/{LANG}/'):
    # 遍历当前目录下的所有文件
    for file in files:
        # 检查文件扩展名是否为JSON
        if file.endswith('.jsonl'):
            file_path = os.path.join(root, file)
            solve(model,file_path)



import pickle
pickle.dump(res,open(f"./data/MIX_{LANG}.pt","wb"))
