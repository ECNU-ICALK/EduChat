#!/usr/bin/env python3
import argparse
import time

import torch
import transformers
from model_training.custom_datasets.formatting import QA_SPECIAL_TOKENS, format_pairs, format_system_prefix
from model_training.models import get_specific_model
from model_training.utils import _strtobool
from tokenizers import pre_tokenizers

from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from transformers.generation.utils import logger
from huggingface_hub import snapshot_download
import mdtex2html
import gradio as gr
import argparse
import warnings
import torch
import os


logger.setLevel("ERROR")
warnings.filterwarnings("ignore")

import warnings

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, required=True)
parser.add_argument("--max_new_tokens", type=int, default=200)
parser.add_argument("--top_k", type=int, default=40)
parser.add_argument("--do-sample", type=_strtobool, default=True)
parser.add_argument("--format", type=str, default="v2")
parser.add_argument("--8bit", action="store_true", dest="eightbit")
parser.add_argument("--system_prefix", type=str, default=None)
parser.add_argument("--per-digit-tokens", action="store_true")


system_prefix = None

args = parser.parse_args()


print('Loading model...')
if args.eightbit:
    model = get_specific_model(
        args.model_path,
        load_in_8bit=True,
        device_map="auto",
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        offload_state_dict=True,
    )
else:
    model = get_specific_model(args.model_path)

model.half().cuda()
model.gradient_checkpointing_enable()  # reduce number of stored activations

print('Loading tokenizer...')
tokenizer = transformers.LlamaTokenizer.from_pretrained(args.model_path)
from model_training.custom_datasets.formatting import QA_SPECIAL_TOKENS

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

    batch = tokenizer.encode(
        format_system_prefix(system_prefix, tokenizer.eos_token)
        if system_prefix
        else ""
        + "".join(format_pairs(conversation_history, tokenizer.eos_token, add_initial_reply_token=True)),
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