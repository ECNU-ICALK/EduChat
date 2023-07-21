from fastapi import FastAPI, Request
from transformers import AutoTokenizer, AutoModel
import uvicorn
import json
import datetime
import torch
from fastapi.middleware.cors import CORSMiddleware
from transformers import LlamaForCausalLM, LlamaTokenizer
import requests
import re

session = requests.Session()
# 正则提取摘要和链接
title_pattern = re.compile('<a.target=..blank..target..(.*?)</a>')
brief_pattern = re.compile('K=.SERP(.*?)</p>')
link_pattern = re.compile(
    '(?<=(a.target=._blank..target=._blank..href=.))(.*?)(?=(..h=))')

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.61 Safari/537.36 Edg/94.0.992.31'}
proxies = {"http": None, "https": None, }


DEVICE = "cuda"
DEVICE_ID = "0"
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE


def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(CUDA_DEVICE):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def find(search_query, page_num=3):
    url = 'https://cn.bing.com/search?q={}'.format(search_query)
    res = session.get(url, headers=headers, proxies=proxies)
    r = res.text

    title = title_pattern.findall(r)
    brief = brief_pattern.findall(r)
    link = link_pattern.findall(r)

    # 数据清洗
    clear_brief = []
    for i in brief:
        tmp = re.sub('<[^<]+?>', '', i).replace('\n', '').strip()
        tmp1 = re.sub('^.*&ensp;', '', tmp).replace('\n', '').strip()
        tmp2 = re.sub('^.*>', '', tmp1).replace('\n', '').strip()
        clear_brief.append(tmp2)

    clear_title = []
    for i in title:
        tmp = re.sub('^.*?>', '', i).replace('\n', '').strip()
        tmp2 = re.sub('<[^<]+?>', '', tmp).replace('\n', '').strip()
        clear_title.append(tmp2)
    return [{'title': "["+clear_title[i]+"]("+link[i][1]+")", 'content':clear_brief[i]}
            for i in range(min(page_num, len(brief)))]


@app.post("/demo")
async def test_domo(request: Request):
    return {}


@app.post("/chat")
async def chat(request: Request):
    global model, tokenizer
    json_post_raw = await request.json()
    print(json_post_raw)
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    functionUsed = json_post_list.get('functionUsed')
    # prompt = json_post_list.get('prompt')
    messages = json_post_list.get('messages')
    max_length = json_post_list.get('max_tokens')
    model_name = json_post_list.get('model')
    temperature = json_post_list.get('temperature')
    top_p = json_post_list.get('top_p')

    prompt = messages[-1]["content"]

    def talk(history, human_input, max_length, temperature):
        prefix = ""
        from enum import Enum

        class ChatRole(str, Enum):
            system = "<|system|>"
            prompter = "<|prompter|>"
            assistant = "<|assistant|>"
        global model, tokenizer
        histories = []
        for question, answer in history:
            histories.append(
                f"{ChatRole.prompter}{question.strip('</s>')}</s>"
                + f"{ChatRole.assistant}{answer.strip('</s>')}</s>"
            )
        if len(histories) > 0:
            prefix += "".join(histories)
            # add sep at the end
        prefix += f"{ChatRole.prompter}{human_input}</s>{ChatRole.assistant}"
        print(prefix)
        inputs = tokenizer(prefix, return_tensors="pt", padding=True).to(0)
        if "token_type_ids" in inputs:
            del inputs["token_type_ids"]
        outputs = model.generate(
            **inputs,
            early_stopping=True,
            max_new_tokens=max_length,
            # do_sample=args.do_sample,
            num_beams=1,
            # top_k=args.top_k,
            top_p=0.7,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id,
            # repetition_penalty=1.01,
        )
        output = tokenizer.decode(outputs[0], truncate_before_pattern=[
                                  r"\n\n^#", "^'''", "\n\n\n"])
        answer = output.split(f"{ChatRole.assistant}")[-1]
        return answer

    history = []
    for i in range(max(-11, -len(messages)+1), -1, 2):
        history.append((messages[i]['content'], messages[i+1]['content']))

    response = talk(history, prompt, max_length if max_length else 2048,
                    temperature if temperature else 0.95)
    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    answer = {
        "response": response,
        # "history": history,
        "status": 200,
        "time": time
    }
    print(answer)
    # log = "[" + time + "] " + '", prompt:"' + prompt + '", response:"' + repr(response) + '"'
    # print(log)
    # torch_gc()
    return answer


if __name__ == '__main__':

    tokenizer = LlamaTokenizer.from_pretrained("edunlp/educhat-002-7b/")
    model = LlamaForCausalLM.from_pretrained(
        "edunlp/educhat-002-7b/", torch_dtype=torch.float16,).cuda()

    model.eval()
    uvicorn.run(app, host='0.0.0.0', port=8888, workers=1)
