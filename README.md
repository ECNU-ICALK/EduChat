# EduChat
<p align="center" width="100%">
<a href="https://www.educhat.top/" target="_blank"><img src="https://github.com/icalk-nlp/EduChat/blob/main/imgs/EduChat.jpeg" alt="EduChat" style="width: 50%; min-width: 300px; display: block; margin: auto;"></a>
</p>

[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/LianjiaTech/BELLE/blob/main/LICENSE)
[![Data License](https://img.shields.io/badge/Data%20License-CC%20BY--NC%204.0-blue.svg)](https://github.com/LianjiaTech/BELLE/blob/main/LICENSE)
[![Generic badge](https://img.shields.io/badge/🤗-Huggingface%20Repo-577CF6.svg)](https://huggingface.co/ecnu-icalk)

[[中文版](https://github.com/icalk-nlp/EduChat/blob/main/README.md)] [[English](https://github.com/icalk-nlp/EduChat/blob/main/README.md)]

## 目录

- [开源清单](#spiral_notepad-开源清单)
  - [模型](#模型)
  - [数据](#数据)
- [介绍](#fountain_pen-介绍)
- [本地部署](#robot-本地部署)
  - [硬件要求](#硬件要求)
  - [下载安装](#下载安装)
  - [使用示例](#使用示例)
- [未来计划](#construction-未来计划)
- [开源协议](#page_with_curl-开源协议)

----

## :spiral_notepad: 开源清单

### 模型

经过educhat-sft-002-data-osm训练后，在我们构建的教育领域多技能高质量数据上继续微调后得到，有7b和13b两个版本可选：

- [**educhat-sft-002-7b**](https://huggingface.co/ecnu-icalk/educhat-sft-002-7b)
- [**educhat-sft-002-13b**](https://huggingface.co/ecnu-icalk/educhat-sft-002-13b)

### 数据

- [**educhat-sft-002-data-osm**](https://huggingface.co/datasets/ecnu-icalk/educhat-sft-002-data-osm): 混合多个开源中英指令、对话数据，并去重后得到。


## :fountain_pen: 介绍

教育是影响人的身心发展的社会实践活动，旨在把人所固有的或潜在的素质自内而外激发出来。因此，必须贯彻“以人为本”的教育理念，重点关注人的个性化、引导式、身心全面发展。为了更好地助力”以人为本“的教育，华东师范大学计算机科学与技术学院的[EduNLP团队](https://www.educhat.top/#/)探索了针对教育垂直领域的对话大模型[EduChat](https://www.educhat.top)相关项目研发。该项目主要研究以预训练大模型为基底的教育对话大模型相关技术，融合多样化的教育垂直领域数据，辅以指令微调、价值观对齐等方法，提供教育场景下自动出题、作业批改、情感支持、课程辅导、高考咨询等丰富功能，服务于广大老师、学生和家长群体，助力实现因材施教、公平公正、富有温度的智能教育。

**开放问答**：

![image](https://github.com/icalk-nlp/EduChat/blob/main/imgs/example_chatedu.gif)

<details><summary><b>作文批改</b></summary>


![image](https://github.com/icalk-nlp/EduChat/blob/main/imgs/example_chatedu.gif)

</details>

<details><summary><b>心理诊断</b></summary>


![image](https://github.com/icalk-nlp/EduChat/blob/main/imgs/example_chatedu.gif)

</details>

<details><summary><b>心理疏导</b></summary>


![image](https://github.com/icalk-nlp/EduChat/blob/main/imgs/example_chatedu.gif)

</details>


## :robot: 本地部署

### 下载安装
1. 下载本仓库内容至本地/远程服务器

```bash
git clone https://github.com/icalk-nlp/EduChat.git
cd EduChat
```

2. 创建conda环境

```bash
conda create --name educhat python=3.8
conda activate educhat
```

3. 安装依赖

```bash
# 首先安装pytorch，安装方法请自行百度。
# 然后安装最新版本的transformers
pip install transformers
```

### 使用示例

#### 单卡部署

以下是一个简单的调用`educhat-sft-002-7b`生成对话的示例代码，可在单张A100/A800或CPU运行，使用FP16精度时约占用15GB显存：

```python
>>> from transformers import LlamaForCausalLM, LlamaTokenizer
>>> tokenizer = LlamaTokenizer.from_pretrained("ecnu-icalk/educhat-sft-002-7b")
>>> model = LlamaForCausalLM.from_pretrained("ecnu-icalk/educhat-sft-002-7b",torch_dtype=torch.float16,).half().cuda()
>>> model = model.eval()

>>> query = "<|prompter|>你好</s><|assistant|>"
>>> inputs = tokenizer(query, return_tensors="pt", padding=True).to(0)
>>> outputs = model.generate(**inputs, do_sample=True, temperature=0.7, top_p=0.8, repetition_penalty=1.02, max_new_tokens=256)
>>> response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
>>> print(response)
您好！我是EduChat，有什么我可以帮助您的吗？ 

>>> query = query + response + "</s><|prompter|>:给我推荐几本心理相关的书籍</s><|assistant|>:"
>>> inputs = tokenizer(query, return_tensors="pt", padding=True).to(0)
>>> outputs = model.generate(**inputs, do_sample=True, temperature=0.7, top_p=0.8, repetition_penalty=1.02, max_new_tokens=256)
>>> response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
>>> print(response)
当然，以下是一些关于心理学的经典书籍：
1.《人性的弱点》（Dale Carnegie）：这本书是一本经典的人际关系指南，它介绍了如何与人相处、如何影响他人以及如何建立良好的人际关系。

2.《心理学与生活》（Richard J. Gerrig）：这本书是一本介绍心理学的入门读物，它涵盖了各种主题，包括认知、情感、人格和社会心理学。

3.《情绪智商》（Daniel Goleman）：这本书介绍了情绪智商的概念，并探讨了情绪智商如何影响我们的生活和工作。

4.《人性的弱点2》（Dale Carnegie）：这本书是《人性的弱点》的续集，它提供了更多的技巧和策略，帮助读者更好地与人相处。

5.《心理学导论》（David G. Myers）：这本书是一本广泛使用的心理学教材，它涵盖了各种主题，包括感知、记忆、思维、情感和人格。
希望这些书籍能够帮助您更深入地了解心理学。
```

#### 网页Demo

**Gradio**

您可以运行本仓库中的[demo/educhat_gradio.py](https://github.com/icalk-nlp/EduChat/blob/main/demo/educhat_gradio.py)：

```bash
python educhat_gradio.py
```

#### Api Demo

你可以运行仓库中的`educhat_api.py`来对外提供一个简单的api服务

```bash
python educhat_api.py
```

启动api服务后，您可以通过网络调用来与EduChat交互

```bash
## curl EduChat
curl -X POST "http://localhost:19324" \
     -H 'Content-Type: application/json' \
     -d '{"prompt": "你是谁？"}'
```

首次调用，您会得到一个api服务返回的uid

```json
{"response":"\n<|Worm|>: 你好，有什么我可以帮助你的吗？","history":[["你好","\n<|Worm|>: 你好，有什么我可以帮助你的吗？"]],"status":200,"time":"2023-04-28 09:43:41","uid":"10973cfc-85d4-4b7b-a56a-238f98689d47"}
```

您可以在后续的对话中填入该uid来和EduChat进行多轮对话

```bash
## curl EduChat multi-round
curl -X POST "http://localhost:19324" \
     -H 'Content-Type: application/json' \
     -d '{"prompt": "你是谁？", "uid":"10973cfc-85d4-4b7b-a56a-238f98689d47"}'
```

#### 命令行Demo

您可以运行仓库中的`educhat_cli_demo.py`来启动一个简单的命令行Demo：

```bash
python educhat_cli_demo.py
```

您可以在该Demo中与EduChat进行多轮对话，输入 `clear` 可以清空对话历史，输入 `stop` 终止Demo。该命令默认使用`educhat-002-7b`单卡运行，您也可以通过参数指定其他模型以及多卡并行，例如：

```bash
python educhat_cli_demo.py --model_name edunlp/educhat-002-13b --gpu 0,1
```

![image](https://github.com/ICALK/EduChat/blob/main/examples/example_educhat_cli_demo.png)


## :construction: 未来计划

从EduChat-001到EduChat-002的迭代过程中，我们逐步增强了它的中文能力、忠实度、安全度，并增加了使用插件的能力。但EduChat-002仍是非常早期的一个模型，我们的旅程也才刚刚开始。未来，我们将持续投入对基础模型的研究，不断开源更加强大的EduChat。

- **课程辅导**：逻辑推理能力是衡量大模型性能的重要指标，我们将通过增大语言模型基座、增强特定训练数据等手段强化EduChat的逻辑推理能力；
- **循循善诱**：语言模型普遍存在幻觉问题和安全性问题，严重阻碍了其实际应用，我们计划在后续版本中继续提高其安全性和可信性。

- **个性化辅导**：我们将逐步将语音、图像等模态深度融入EduChat，使其具备跨模态理解和生成能力；
- **工具调用**：我们期望的EduChat应当是千人千面的，未来我们希望能够给每个人一个独一无二的EduChat，它将在与你的交互中持续学习，伴随你的成长而成长，成为你的专属助手。


## :page_with_curl: 开源协议、模型局限、使用限制与免责声明

本项目所含代码采用[Apache 2.0](https://github.com/ICALK/EduChat/blob/main/LICENSE)协议，数据采用[CC BY-NC 4.0](https://github.com/ICALK/EduChat/blob/main/DATA_LICENSE)协议。

尽管我们对EduChat进行了优化，但仍存在以下问题，需要进行改进：

- 当涉及到事实性指令时，可能会产生错误的回答，与实际事实相悖。

- 模型回复可能存在偏见，有可能生成危险性言论。

- 在某些场景中，比如推理、代码、多轮对话等方面，模型的能力仍有待提高。

鉴于上述模型的局限性，我们要求开发者仅将我们开源的代码、数据、模型以及由该项目生成的衍生物用于研究目的，禁止用于商业用途，以及其他可能对社会带来危害的用途。

本项目仅供研究目的使用，项目开发者对于使用本项目（包括但不限于数据、模型、代码等）所导致的任何危害或损失不承担责任。详情请参考该[免责声明](https://github.com/icalk-nlp/EduChat/blob/main/LICENSE/DISCLAIMER)。

## :heart: 致谢

- [LLaMa](https://arxiv.org/abs/2302.13971): 基座模型在LLaMa基础上使用中英文数据预训练
- [华东师范大学出版社](https://www.ecnupress.com.cn/)：数据支持
- [竹蜻蜓数据科技（浙江）有限公司](https://www.autopaddle.com//): 开发支持
- [邱锡鹏教授](https://xpqiu.github.io/): 项目顾问