#!/usr/bin/env python3
import argparse

import torch
import transformers
from distutils.util import strtobool
from tokenizers import pre_tokenizers

from transformers.generation.utils import logger
import mdtex2html
import gradio as gr
import argparse
import warnings
import torch


logger.setLevel("ERROR")
warnings.filterwarnings("ignore")

import warnings

warnings.filterwarnings("ignore")

def _strtobool(x):
    return bool(strtobool(x))

QA_SPECIAL_TOKENS = {
    "Question": "<|prompter|>",
    "Answer": "<|assistant|>",
    "System": "<|system|>",
    "StartPrefix": "<|prefix_begin|>",
    "EndPrefix": "<|prefix_end|>",
    "InnerThought":"<|inner_thoughts|>",
    "EndOfThought":"<eot>"
}

def format_pairs(pairs, eos_token, add_initial_reply_token=False):
    conversations = [
        "{}{}{}".format(QA_SPECIAL_TOKENS["Question" if i % 2 == 0 else "Answer"], pairs[i], eos_token)
        for i in range(len(pairs))
    ]
    if add_initial_reply_token:
        conversations.append(QA_SPECIAL_TOKENS["Answer"])
    return conversations

def format_system_prefix(prefix, eos_token):
    return "{}{}{}".format(
        QA_SPECIAL_TOKENS["System"],
        prefix,
        eos_token,
    )

def get_specific_model(
    model_name, seq2seqmodel=False, without_head=False, cache_dir=".cache", quantization=False, **kwargs
):
    # encoder-decoder support for Flan-T5 like models
    # for now, we can use an argument but in the future,
    # we can automate this

    model = transformers.LlamaForCausalLM.from_pretrained(model_name, **kwargs)

    return model

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, required=True)
parser.add_argument("--max_new_tokens", type=int, default=200)
parser.add_argument("--top_k", type=int, default=40)
parser.add_argument("--do_sample", type=_strtobool, default=True)
# parser.add_argument("--system_prefix", type=str, default=None)
parser.add_argument("--per-digit-tokens", action="store_true")


args = parser.parse_args()

# # 开放问答
# system_prefix = \
# "<|system|>"'''你是一个人工智能助手，名字叫EduChat。
# - EduChat是一个由华东师范大学开发的对话式语言模型。
# EduChat的工具
# - Web search: Disable.
# - Calculators: Disable.
# EduChat的能力
# - Inner Thought: Disable.
# 对话主题
# - General: Enable.
# - Psychology: Disable.
# - Socrates: Disable.'''"</s>"

# # 启发式教学
# system_prefix = \
# "<|system|>"'''你是一个人工智能助手，名字叫EduChat。
# - EduChat是一个由华东师范大学开发的对话式语言模型。
# EduChat的工具
# - Web search: Disable.
# - Calculators: Disable.
# EduChat的能力
# - Inner Thought: Disable.
# 对话主题
# - General: Disable.
# - Psychology: Disable.
# - Socrates: Enable.'''"</s>"

# 情感支持
system_prefix = \
"<|system|>"'''你是一个人工智能助手，名字叫EduChat。
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

# # 情感支持(with InnerThought)
# system_prefix = \
# "<|system|>"'''你是一个人工智能助手，名字叫EduChat。
# - EduChat是一个由华东师范大学开发的对话式语言模型。
# EduChat的工具
# - Web search: Disable.
# - Calculators: Disable.
# EduChat的能力
# - Inner Thought: Enable.
# 对话主题
# - General: Disable.
# - Psychology: Enable.
# - Socrates: Disable.'''"</s>"



print('Loading model...')

model = get_specific_model(args.model_path)

model.half().cuda()
model.gradient_checkpointing_enable()  # reduce number of stored activations

print('Loading tokenizer...')
tokenizer = transformers.LlamaTokenizer.from_pretrained(args.model_path)

tokenizer.add_special_tokens(
            {
                "pad_token": "</s>",
                "eos_token": "</s>",
                "sep_token": "<s>",
            }
        )
additional_special_tokens = (
        []
        if "additional_special_tokens" not in tokenizer.special_tokens_map
        else tokenizer.special_tokens_map["additional_special_tokens"]
    )
additional_special_tokens = list(set(additional_special_tokens + list(QA_SPECIAL_TOKENS.values())))

print("additional_special_tokens:", additional_special_tokens)

tokenizer.add_special_tokens({"additional_special_tokens": additional_special_tokens})

if args.per_digit_tokens:
    tokenizer._tokenizer.pre_processor = pre_tokenizers.Digits(True)

human_token_id = tokenizer.additional_special_tokens_ids[
    tokenizer.additional_special_tokens.index(QA_SPECIAL_TOKENS["Question"])
]

print('Type "quit" to exit')
print("Press Control + C to restart conversation (spam to exit)")

conversation_history = []


"""Override Chatbot.postprocess"""


def postprocess(self, y):
    if y is None:
        return []
    for i, (message, response) in enumerate(y):
        y[i] = (
            None if message is None else mdtex2html.convert((message)),
            None if response is None else mdtex2html.convert(response),
        )
    return y


gr.Chatbot.postprocess = postprocess


def parse_text(text):
    """copy from https://github.com/GaiZhenbiao/ChuanhuChatGPT/"""
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f'<br></code></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", "\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>"+line
    text = "".join(lines)
    return text


def predict(input, chatbot, max_length, top_p, temperature, history):
    query = parse_text(input)
    chatbot.append((query, ""))
    conversation_history = []
    for i, (old_query, response) in enumerate(history):
        conversation_history.append(old_query)
        conversation_history.append(response)

    conversation_history.append(query)

    query_str = "".join(format_pairs(conversation_history, tokenizer.eos_token, add_initial_reply_token=True))

    if system_prefix:
        query_str = system_prefix + query_str
    print("query:", query_str)

    batch = tokenizer.encode(
        query_str,
        return_tensors="pt",
    )

    with torch.cuda.amp.autocast():
        out = model.generate(
            input_ids=batch.to(model.device),
            max_new_tokens=args.max_new_tokens, # The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt.
            do_sample=args.do_sample,
            max_length=max_length,
            top_k=args.top_k,
            top_p=top_p,
            temperature=temperature,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    if out[0][-1] == tokenizer.eos_token_id:
        response = out[0][:-1]
    else:
        response = out[0]

    response = tokenizer.decode(out[0]).split(QA_SPECIAL_TOKENS["Answer"])[-1]

    conversation_history.append(response)

    with open("./educhat_query_record.txt", 'a+') as f:
        f.write(str(conversation_history) + '\n')


    chatbot[-1] = (query, parse_text(response))
    history = history + [(query, response)]
    print(f"chatbot is {chatbot}")
    print(f"history is {history}")

    return chatbot, history


def reset_user_input():
    return gr.update(value='')


def reset_state():
    return [], []


with gr.Blocks() as demo:
    gr.HTML("""<h1 align="center">欢迎使用 EduChat 人工智能助手！</h1>""")

    chatbot = gr.Chatbot()
    with gr.Row():
        with gr.Column(scale=4):
            with gr.Column(scale=12):
                user_input = gr.Textbox(show_label=False, placeholder="Input...", lines=10).style(
                    container=False)
            with gr.Column(min_width=32, scale=1):
                submitBtn = gr.Button("Submit", variant="primary")
        with gr.Column(scale=1):
            emptyBtn = gr.Button("Clear History")
            max_length = gr.Slider(
                0, 2048, value=2048, step=1.0, label="Maximum length", interactive=True)
            top_p = gr.Slider(0, 1, value=0.2, step=0.01,
                              label="Top P", interactive=True)
            temperature = gr.Slider(
                0, 1, value=1, step=0.01, label="Temperature", interactive=True)

    history = gr.State([])  # (message, bot_message)

    submitBtn.click(predict, [user_input, chatbot, max_length, top_p, temperature, history], [chatbot, history],
                    show_progress=True)
    submitBtn.click(reset_user_input, [], [user_input])

    emptyBtn.click(reset_state, outputs=[chatbot, history], show_progress=True)

demo.queue().launch(inbrowser=True, share=True)