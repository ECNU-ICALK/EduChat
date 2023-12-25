# Standard library imports
import datetime
import heapq
import json
import logging
import re
import time
import warnings
from argparse import ArgumentParser
from copy import deepcopy
from curses.ascii import isdigit
from urllib import parse

# Related third-party imports
import fasttext
import requests
import spacy
import urllib3
from bs4 import BeautifulSoup
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Local application/library specific imports
from score_utils import score, score_2, score_3
from textrank_utils import top_sentence

# Configures
google_api_key = "YOUR_GOOGLE_API_KEY"
warnings.filterwarnings('ignore')

if_answerbox = False
class prey(object):
    def __init__(self, value, sentence):
        self.value =  value
        self.sentence = sentence
    # 重写 < 符号用于sorted
    def __lt__(self, other):
        return self.value < other.value
    def __gt__(self, other):
        return self.value > other.value
    def __le__(self, other):
        return self.value <= other.value
    def __eq__(self, other):
        return self.value == other.value
    def __ne__(self, other):
        return self.value != other.value
    def __ge__(self, other):
        return self.value >= other.value

def containenglish(str0):
    import re
    return bool(re.search('[a-z]', str0))


def jdump(obj, f, mode="w", indent=4, default=str):
    """Dump a str or dictionary to a file in json format.

    Args:
        obj: An object to be written.
        f: A string path to the location on disk.
        mode: Mode for opening the file.
        indent: Indent for storing json dictionaries.
        default: A function to handle non-serializable entries; defaults to `str`.
    """
    f = open(f, mode='w', encoding='utf-8')
    if isinstance(obj, (dict, list)):
        json.dump(obj, f, indent=indent, ensure_ascii=False)
    elif isinstance(obj, str):
        f.write(obj)
    else:
        raise ValueError(f"Unexpected type: {type(obj)}")
    f.close()

def clean_html(html: str) -> str:
    """Remove HTML markup from the given string."""
    # Remove inline JavaScript/CSS, HTML comments, and HTML tags
    cleaned_html = re.sub(
        r"(?is)<(script|style).*?>.*?(</\1>)|<!--(.*?)-->[\n]?|<(?s).*?>", "", html.strip()
    )

    # Deal with whitespace and HTML entities
    cleaned_html = re.sub(
        r"&nbsp;|  |\t|&.*?;[0-9]*&.*?;|&.*?;", "", cleaned_html
    )

    return cleaned_html.strip()

def select(new):
    if len(new) < 10:
        oral = new
    elif len(new) // 10 < 10:
        oral = new[:20]
    elif len(new) // 10 > 50:
        oral = new[:50]
    else:
        oral = new[:len(new) // 10] 
    return oral

def get_web_response(url):
    try:
        response = requests.get(url=url, timeout=4)
        response.encoding = 'utf-8'
        return response
    except requests.exceptions.RequestException:
        print("requests post fail")
        return None

def extract_description(soup):
    description = soup.find(attrs={"name": "description"})
    if description:
        content = description.get('content')
        if content:
            return content
    return None

def summ_web(q, url, ft_en, ft_zh, is_eng, nlp_en, nlp_zh, measure_en, measure_zh, snippet,title):
    url = parse.unquote(url)

    response = None
    if response is None:
        return {"title":title, "url": url, "summ": snippet, "note": "fail to get ... use snippet", "type": "snippet"}

    soup = BeautifulSoup(response.text, "html.parser")
    description = extract_description(soup)

    if description:
        if all(key_word in description for key_word in q.split()):
            return {"title":title, "url": url, "summ": description, "note": "use description as summ", "type": "description"}

    text = clean_html(response.text)
    sentences = re.split("\n|。|\.", text)

    ft = ft_en if is_eng else ft_zh
    measure = measure_en if is_eng else measure_zh
    nlp = nlp_en if is_eng else nlp_zh

    scored_sentences = []
    for sentence in sentences:
        if 3 <= len(sentence) <= 200:
            scored_sentence = {
                'ft': -1 * score(q, sentence, ft) if ft else None,
                'score_2': -1 * score_2(q, sentence),
                'measure': -1 * score_3(q, sentence, measure=measure) if measure else None,
                'sentence': sentence
            }
            scored_sentences.append(scored_sentence)

    top_sentences = heapq.nsmallest(5, scored_sentences, key=lambda x: x['ft'] or float('inf')) + \
                    heapq.nsmallest(10, scored_sentences, key=lambda x: x['score_2']) + \
                    heapq.nsmallest(5, scored_sentences, key=lambda x: x['measure'] or float('inf'))

    stop_word = "." if is_eng else "。"
    combined_text = stop_word.join([sentence['sentence'] for sentence in top_sentences])

    if len(combined_text) < 3:
        return {"title":title, "url": url, "summ": snippet, "note": "bad web, fail to summ, use snippet,", "type": "snippet"}

    try:
        summary = top_sentence(text=combined_text, limit=3, nlp=nlp)
        summary = "".join(summary)
    except Exception as e:
        return {"title":title, "url": url, "summ": snippet, "note": "unknown summ error , use snippet", "type": "snippet"}

    if any(key_word in summary for key_word in q.split()):
        return {"title":title, "url": url, "summ": summary, "note": "good summ and use it", "type": "my_summ"}

    return {"title":title, "url": url, "summ": snippet, "note": "poor summ , use snippet", "type": "snippet"}
    
def search_api(q, SERPER_KEY):
    import requests
    import json
    url = "https://google.serper.dev/search"

    if containenglish(q): 
        payload = json.dumps({"q": q,})
    else:
        payload = json.dumps({"q": q})#,"gl": "cn","hl": "zh-cn"})
    headers = {
        'X-API-KEY': SERPER_KEY,
        'Content-Type': 'application/json'
    }
    logging.captureWarnings(True)
    urllib3.disable_warnings()
    requests.adapters.DEFAULT_RETRIES = 5
    response = requests.request("POST", url, headers=headers, data=payload, verify=False)
    response.keep_alive = False

    response_dict = json.loads(response.text)

    return response_dict

def filter_urls(urls, snippets, titles, black_list=None, topk=3):
    if black_list is None:
        black_list = ["enoN, youtube.com, bilibili.com", "zhihu.com"]

    filtered_urls, filtered_snippets, filtered_titles = [], [], []
    count = 0
    for url, snippet, title in zip(urls, snippets, titles):
        if all(domain not in url for domain in black_list) and url.split(".")[-1] != "pdf":
            filtered_urls.append(url)
            filtered_snippets.append(snippet)
            filtered_titles.append(title)
            count += 1
            if count >= topk:
                break

    return filtered_urls, filtered_snippets, filtered_titles

def engine(q, SERPER_KEY,ft_en, ft_zh, nlp_en, nlp_zh, measure_en, measure_zh, topk=3):
    global if_answerbox
    start_time = time.time()
    is_eng = containenglish(q)

    response = search_api(q, SERPER_KEY)

    if "answerBox" in response.keys():
        url = response["answerBox"].get("link", response["organic"][0]["link"])
        summ = response["answerBox"]
        print("[EnGINE] answerBox")
        print("[ENGINE] query cost:", time.time() - start_time)

        if_answerbox = True
        return {"url": url, "summ": summ, "note": "directly return answerBox, thx google !", "type": "answerBox"}

    raw_urls = [i["link"] for i in response["organic"]]
    raw_snippets = [i["snippet"] for i in response["organic"]]
    raw_titles = [i["title"] for i in response["organic"]]
    urls, snippets, titles = filter_urls(raw_urls, raw_snippets, raw_titles, topk=topk)

    results = {}
    for i, url in enumerate(urls):
        try:
            summ = summ_web(q, url, ft_en, ft_zh, is_eng, nlp_en, nlp_zh, measure_en, measure_zh, snippets[i], titles[i])
        except:
            summ = {"url": url, "summ": snippets[i], "note": "unbelievable error, use snippet !", "type": "snippet", "title":titles[i]}

        results[str(i)] = summ

    return results   

def search(text):
    global nlp_en,nlp_zh,ft_en,ft_zh,measure_en,measure_zh,google_api_key
    PROMPT_ASK = '''[QUESTION]
###检索内容：
[SEARCH]'''
    ask = PROMPT_ASK.replace("[QUESTION]",text)
    global if_answerbox
    temps=engine(text, google_api_key, ft_en, ft_zh, nlp_en, nlp_zh, measure_en, measure_zh)
    data = dict()
    d = data.setdefault('similar_qa', [])
    if if_answerbox:
        qa = dict()
        qa['similar_sentence'] = temps['summ']['title']
        if 'snippet' in temps['summ']:
            qa['answer'] = temps['summ']['snippet']
        else:
            qa['answer'] = temps['summ']['answer']
        qa['url'] = temps['url']
        d.append(qa)
        if_answerbox = False
    else:
        for temp in temps.keys():
            qa = dict()
            qa['similar_sentence'] = temps[temp]['title']
            qa['answer'] = temps[temp]['summ']
            qa['url'] = temps[temp]['url']
            d.append(qa)
    _search = ""
    for i in range(len(d)):
        _search += f"（{str(i+1)}）" + d[i]["similar_sentence"] + "：" + d[i]["answer"] + "\n"
    ask = ask.replace("[SEARCH]",_search)
    return ask, d

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/chat")
async def chat(request: Request):
    global model, tokenizer,rule
    json_post_raw = await request.json()
    print(json_post_raw)
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    functionUsed = json_post_list.get('functionUsed')
    messages = json_post_list.get('messages')
    messages_copy = deepcopy(messages)
    print("messages:", messages)
    max_length = json_post_list.get('max_tokens')
    model_name = json_post_list.get('model')
    temperature = json_post_list.get('temperature')
    top_p = json_post_list.get('top_p')

    prompt = messages[-1]["content"]
    def talk(history,human_input,max_length,temperature,functionUsed):
        if functionUsed == "teacher-guide":
            # 启发式教学
            prefix = "<|system|>"'''你是一个人工智能助手，名字叫EduChat。
- EduChat是一个由华东师范大学开发的对话式语言模型。
EduChat的工具
- Web search: Disable.
- Calculators: Disable.
EduChat的能力
- Inner Thought: Disable.
对话主题
- General: Disable.
- Psychology: Disable.
- Socrates: Enable.'''"</s>"
        elif functionUsed == "emotionEase":
            # 心理
            prefix = "<|system|>"'''你是一个人工智能助手，名字叫EduChat。
- EduChat是一个由华东师范大学开发的对话式语言模型。
EduChat的工具
- Web search: Disable.
- Calculators: Disable.
EduChat的能力
- Inner Thought: Disable.
对话主题
- General: Disable.
- Psychology: Enable.
- Socrates: Disable.'''"</s>"        
        elif functionUsed == "retrievalQA":
            # 搜索
            prefix = "<|system|>"'''你是一个人工智能助手，名字叫EduChat。
- EduChat是一个由华东师范大学开发的对话式语言模型。
EduChat的工具
- Web search: Enable.
- Calculators: Disable.
EduChat的能力
- Inner Thought: Enable.
对话主题
- General: Enable.
- Psychology: Disable.
- Socrates: Disable.'''"</s>"        
        else:
            # OpenDomain
            prefix = "<|system|>"'''你是一个人工智能助手，名字叫EduChat。
- EduChat是一个由华东师范大学开发的对话式语言模型。
EduChat的工具
- Web search: Disable.
- Calculators: Disable.
EduChat的能力
- Inner Thought: Disable.
对话主题
- General: Enable.
- Psychology: Disable.
- Socrates: Disable.'''"</s>"
        from enum import Enum
        class ChatRole(str, Enum):
            system = "<|system|>"
            prompter = "<|prompter|>"
            assistant = "<|assistant|>"
        global model, tokenizer,llm,sampling_params
        histories = []
        for question, answer in history:
            histories.append(
                f"{ChatRole.prompter}{question.strip('</s>')}</s>"
                + f"{ChatRole.assistant}{answer.strip('</s>')}</s>"
            )
        for i in range(len(histories),-1,-1):
            suppose = prefix+"".join(histories[i:])+f"{ChatRole.prompter}{human_input}</s>{ChatRole.assistant}"
            if len(tokenizer.tokenize(suppose)) > 2048-512:
                histories=histories[i+1:]
                break
        if len(histories) > 0:
            prefix += "".join(histories)
            # add sep at the end
        if functionUsed == "retrievalQA":
            text,d = search(human_input)
            prefix += f"{ChatRole.prompter}{text}</s>{ChatRole.assistant}"
        else:
            prefix += f"{ChatRole.prompter}{human_input}</s>{ChatRole.assistant}"

        outputs = llm.generate([prefix], sampling_params)
        answer = outputs[0].outputs[0].text.split(f"{ChatRole.assistant}")[-1].strip("None").strip()
    
        if functionUsed == "retrievalQA":
            ppos = 0
            if answer.count("自己的知识来回答。"):
                ppos = max(ppos,answer.find("自己的知识来回答。")+len("自己的知识来回答。"))
            if answer.count("<eot>"):
                ppos = max(ppos,answer.find("<eot>")+len("<eot>"))
            if answer.count("<|inner_thoughts|>"):
                ppos = max(ppos,answer.find("<|inner_thoughts|>")+len("<|inner_thoughts|>"))
            inner = answer[:ppos]
            answer = re.sub("<\|inner_thoughts\|>.*?<eot>", "", answer, flags=re.DOTALL)
            def clean(s,t):
                if s.find(t)==-1:
                    return s
                pos = s.find(t)+len(t)
                return s[pos:]
                
            answer = clean(answer,"自己的知识来回答。")
            answer = clean(answer,"<eot>")
            answer = clean(answer,"<|inner_thoughts|>")
            ref = ""
            if inner.count("所有的相关信息对回答问题是有帮助的"):
                
                for i in range(len(d)):
                    ref += f'- [{d[i]["similar_sentence"]}]({d[i]["url"]})\n'
            elif inner.count("所以我可以利用相关信息"):
                l = []
                for i in range(inner.find("所以我可以利用相关信息")+5,len(inner)):
                    if isdigit(inner[i]):
                        l.append(int(inner[i])-1)
                for idx, i in enumerate(l):
                    ref += f'- [{d[i]["similar_sentence"]}]({d[i]["url"]})\n'
            if len(ref):
                answer += "\n\n\n##### 参考链接：\n" + ref + "\n"
            answer = answer.strip()
        return answer

    history = []
    for i in range(max(-16, -len(messages)+1), -1, 2):
        history.append((messages[i]['content'], messages[i+1]['content']))
    response = talk(history,prompt,max_length if max_length else 2048,temperature if temperature else 0.95,functionUsed)
    messages_copy.append({"role": "system", "content":response})
    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")

    answer = {
        "response": response,
        "inner_thought":"", 
        "status": 200,
        "time": time
    }
    
    return answer

def initialize_model_and_tokenizer(args):
    global llm,sampling_params,tokenizer, model
    path = args.checkpoint_path
    from vllm import LLM, SamplingParams
    sampling_params = SamplingParams(temperature=0.8, top_p=0.4,max_tokens=1024)
    llm = LLM(model=path, tensor_parallel_size=1, trust_remote_code=True)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(path)

def load_components():
    global nlp_en,nlp_zh,ft_en,ft_zh,measure_en,measure_zh
    nlp_en = spacy.load("en_core_web_sm")
    nlp_zh = spacy.load("zh_core_web_sm")
    ft_en = fasttext.load_model('cc.en.300.bin')
    ft_zh = fasttext.load_model('cc.zh.300.bin')
    measure_en = None
    measure_zh = None

def _get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "-c",
        "--checkpoint-path",
        type=str,
        default="ecnu-icalk/educhat-sft-002-13b-baichuan",
        help="Checkpoint name or path, default to %(default)r",
    )
    parser.add_argument(
        "--server-port", type=int, default=8000, help="Demo server port."
    )
    parser.add_argument(
        "--server-name",
        type=str,
        default="0.0.0.0",
        help="Demo server name. Default: 127.0.0.1, which is only visible from the local computer."
        " If you want other computers to access your server, use 0.0.0.0 instead.",
    )

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = _get_args()

    initialize_model_and_tokenizer(args)

    load_components()

    uvicorn.run(app, host=args.server_name, port=args.server_port, workers=1)