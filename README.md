# EduChat
教育是影响人的身心发展的社会实践活动，旨在把人所固有的或潜在的素质自内而外激发出来。因此，必须贯彻“以人为本”的教育理念，重点关注人的个性化、引导式、身心全面发展。为了更好地助力”以人为本“的教育，华东师范大学计算机科学与技术学院的[EduNLP团队](https://www.educhat.top/#/)探索了针对教育垂直领域的对话大模型[EduChat](https://www.educhat.top)相关项目研发。该项目主要研究以预训练大模型为基底的教育对话大模型相关技术，融合多样化的教育垂直领域数据，辅以指令微调、价值观对齐等方法，提供教育场景下自动出题、作业批改、情感支持、课程辅导、高考咨询等丰富功能，服务于广大老师、学生和家长群体，助力实现因材施教、公平公正、富有温度的智能教育。

上节课上到哪里了？学生掌握情况怎么样？这节课要上什么内容呢？我要怎么上才能要学生听懂呢？学生可能会存在什么问题呢？为了让学生更好更快地学习，老师在上课之前，往往会先思考“教什么？如何教？”的问题。能不能要大模型也学会先思考再育人（Thinking before teaching）呢？针对这个问题，华东师范大学计算机科学与技术学院的EduNLP团队研发了推理教育大模型EduChat-R1，一个更懂学生、更会教学的教育专用推理大模型。该模型基于该团队最新研发的教育大模型基座EduChat 2.0，构建教育场景特有的深度推理指令数据集，并通过强化学习训练实现模型教学场景慢思考能力涌现。目前，本团队同时开源了EduChat 2.0和EduChat-R1模型，8B、32B等多个版本模型参数已在Github和 Hugging Face 等平台开放，助力大模型在智能教育领域的研究和应用发展。

基于EduChat 2.0以及EduChat-R1模型，团队围绕教育心理、教育安全、数字治疗、辅助授课和教材出版等场景，研发了舒心阁（MindCare@EduChat）、安全加固（Shell@EduChat）、奇迹疗愈（MiracleH@EduChat）、AI智慧黑板（AiBoard@EduChat）和敏捷出版（AgiPub@EduChat）等多款产品。

<p align="center" width="100%">
<a href="https://www.educhat.top/" target="_blank"><img src="https://github.com/icalk-nlp/EduChat/blob/main/imgs/EduChat.jpeg" alt="EduChat" style="width: 50%; min-width: 300px; display: block; margin: auto;"></a>
</p>

[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/LianjiaTech/BELLE/blob/main/LICENSE)
[![Data License](https://img.shields.io/badge/Data%20License-CC%20BY--NC%204.0-blue.svg)](https://github.com/LianjiaTech/BELLE/blob/main/LICENSE)
[![Generic badge](https://img.shields.io/badge/WeChat-EduChat-green.svg?logo=wechat)](https://github.com/icalk-nlp/EduChat/blob/main/imgs/WeChat_EduChat.JPG)
[![Generic badge](https://img.shields.io/badge/🤗-Huggingface%20Repo-577CF6.svg)](https://huggingface.co/ecnu-icalk)

<!-- [[中文版](https://github.com/icalk-nlp/EduChat/blob/main/README.md)] [[English](https://github.com/icalk-nlp/EduChat/blob/main/README.md)] -->
- 体验地址：https://www.educhat.top/ 

## 目录

- [开源清单](#spiral_notepad-开源清单)
- [介绍](#fountain_pen-介绍)
- [引用](#引用)
- [本地部署](#robot-本地部署)
  - [下载安装](#下载安装)
  - [使用示例](#使用示例)
- [未来计划](#construction-未来计划)
- [开源协议](#page_with_curl-开源协议)

----

## :spiral_notepad: 开源清单

### 模型
**注意：使用前按照模型介绍页面中的使用方法部分解密**
- [**EduChat 0.1** (educhat-sft-002-13b-baichuan)](https://huggingface.co/ecnu-icalk/educhat-sft-002-13b-baichuan)：在educhat-base-002-13b-baichuan基础上，使用我们构建的教育领域多技能数据微调后得到
- [**EduChat 0.1** (educhat-base-002-13b-baichuan)]()：使用educhat-sft-002-data-osm数据训练得到
- [**EduChat 0.1** (educhat-sft-002-7b)](https://huggingface.co/ecnu-icalk/educhat-sft-002-7b)：在educhat-base-002-7b基础上，使用我们构建的教育领域多技能数据微调后得到
- [**EduChat 0.1** (educhat-base-002-7b)](https://huggingface.co/ecnu-icalk/educhat-base-002-7b)：使用educhat-sft-002-data-osm数据训练得到
- [**EduChat 0.1** (educhat-sft-002-13b)](https://huggingface.co/ecnu-icalk/educhat-sft-002-13b)：训练方法与educhat-sft-002-7b相同，模型大小升级为13B
- [**EduChat 0.1** (educhat-base-002-13b)](https://huggingface.co/ecnu-icalk/educhat-base-002-13b)：训练方法与educhat-base-002-7b相同，模型大小升级为13B
- [**EduChat 1.0** (educhat-sft-002-1.8b-qwen1.5)](https://huggingface.co/ecnu-icalk/educhat-sft-002-1.8b-qwen1.5)：基于Qwen1.5 1.8B训练得到
- [**EduChat 1.0** (educhat-sft-002-14b-qwen1.5)](https://huggingface.co/ecnu-icalk/educhat-sft-002-14b-qwen1.5)：基于Qwen1.5 14B训练得到
- [**EduChat 1.0** (educhat-sft-002-32b-qwen1.5)](https://huggingface.co/ecnu-icalk/educhat-sft-002-32b-qwen1.5)：基于Qwen1.5 32B训练得到
- [**EduChat 2.0** (educhat-sft-003-32b-qwen2.5)](https://huggingface.co/ecnu-icalk/educhat-sft-003-32b-qwen2.5)：基于Qwen2.5 32B训练得到
- [**EduChat 2.0** (educhat-sft-003-72b-qwen2.5)](https://huggingface.co/ecnu-icalk/educhat-sft-003-72b-qwen2.5)：基于Qwen2.5 72B训练得到
- [**EduChat-R1** (educhat-r1-001-32b-qwen3.0)](https://huggingface.co/ecnu-icalk/educhat-r1-001-32b-qwen3.0)：基于Qwen3.0 32B训练得到
- [**EduChat-R1** (educhat-r1-001-8b-qwen3.0)](https://huggingface.co/ecnu-icalk/educhat-r1-001-8b-qwen3.0)：基于Qwen3.0 8B训练得到

### 数据

- [**educhat-sft-002-data-osm**](https://huggingface.co/datasets/ecnu-icalk/educhat-sft-002-data-osm): 混合多个开源中英指令、对话数据，并去重后得到，约400w

### 代码

数据质量对于模型性能至关重要，为此，我们开源了数据清洗工具CleanTool(可选使用GPU Turbo Speed Up)，包括数据去重，低质量数据删除等功能，未来将继续不断完善。

- [CleanTool](https://github.com/icalk-nlp/EduChat/blob/main/clean_tool
)

## 引用
EduChat: A Large-Scale Language Model-based Chatbot System for Intelligent Education

链接：https://arxiv.org/abs/2308.02773

如果使用本项目的代码、数据或模型，请引用本项目论文：
```
@article{educhat2023,
  title={EduChat: A Large-Scale Language Model-based Chatbot System for Intelligent Education},
  author={Yuhao Dan, Zhikai Lei, Yiyang Gu, Yong Li, Jianghao Yin, Jiaju Lin, Linhao Ye, Zhiyan Tie, Yougen Zhou, Yilei Wang, Aimin Zhou, Ze Zhou, Qin Chen, Jie Zhou, Liang He, Xipeng Qiu},
  journal={CCKS 2024},
  year={2024}
}
```

## :fountain_pen: 介绍

教育是影响人的身心发展的社会实践活动，旨在把人所固有的或潜在的素质自内而外激发出来。因此，必须贯彻“以人为本”的教育理念，重点关注人的个性化、引导式、身心全面发展。为了更好地助力”以人为本“的教育，华东师范大学计算机科学与技术学院的[EduNLP团队](https://www.educhat.top/#/)探索了针对教育垂直领域的对话大模型[EduChat](https://www.educhat.top)相关项目研发。该项目主要研究以预训练大模型为基底的教育对话大模型相关技术，融合多样化的教育垂直领域数据，辅以指令微调、价值观对齐等方法，提供教育场景下自动出题、作业批改、情感支持、课程辅导、高考咨询等丰富功能，服务于广大老师、学生和家长群体，助力实现因材施教、公平公正、富有温度的智能教育。

**基础能力**：

![image](https://github.com/icalk-nlp/EduChat/blob/main/imgs/基础能力.gif)

<details><summary><b>开放问答</b></summary>


![image](https://github.com/icalk-nlp/EduChat/blob/main/imgs/开放问答.gif)

</details>

<details><summary><b>情感支持</b></summary>


![image](https://github.com/icalk-nlp/EduChat/blob/main/imgs/情感支持.gif)

</details>

<details><summary><b>作文批改</b></summary>


![image](https://github.com/icalk-nlp/EduChat/blob/main/imgs/作文批改.gif)

</details>

<details><summary><b>启发式教学</b></summary>


![image](https://github.com/icalk-nlp/EduChat/blob/main/imgs/循循善诱.gif)

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

#### 输入格式

使用EduChat时，sft模型的输入格式为system_prompt + query。根据所需功能不同从以下的system_prompt中选择。base模型在使用时不需要添加system_prompt。

开放问答
```
system_prompt = \
"<|system|>"'''你是一个人工智能助手，名字叫EduChat。
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
```

启发式教学
```
system_prompt = \
"<|system|>"'''你是一个人工智能助手，名字叫EduChat。
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
```

情感支持
```
system_prompt = \
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
```

情感支持(with InnerThought)
```
system_prompt = \
"<|system|>"'''你是一个人工智能助手，名字叫EduChat。
- EduChat是一个由华东师范大学开发的对话式语言模型。
EduChat的工具
- Web search: Disable.
- Calculators: Disable.
EduChat的能力
- Inner Thought: Enable.
对话主题
- General: Disable.
- Psychology: Enable.
- Socrates: Disable.'''"</s>"
```

#### 单卡部署

以下是一个简单的调用`educhat-sft-002-7b`生成对话的示例代码，可在单张A100/A800或CPU运行，使用FP16精度时约占用15GB显存：

```python
>>> from transformers import LlamaForCausalLM, LlamaTokenizer
>>> tokenizer = LlamaTokenizer.from_pretrained("ecnu-icalk/educhat-sft-002-7b")
>>> model = LlamaForCausalLM.from_pretrained("ecnu-icalk/educhat-sft-002-7b",torch_dtype=torch.float16,).half().cuda()
>>> model = model.eval()

>>> query = system_prompt + "<|prompter|>你好</s><|assistant|>"
>>> inputs = tokenizer(query, return_tensors="pt", padding=True).to(0)
>>> outputs = model.generate(**inputs, do_sample=True, temperature=0.7, top_p=0.8, repetition_penalty=1.02, max_new_tokens=256)
>>> response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
>>> print(response)
你好！我是EduChat，有什么我可以帮助你的吗？ 

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
希望这些书籍能够帮助你更深入地了解心理学。
```

#### 网页Demo

**Gradio**

你可以运行本仓库中的[demo/educhat_gradio.py](https://github.com/icalk-nlp/EduChat/blob/main/demo/educhat_gradio.py)：

```bash
python educhat_gradio.py --model_path /path/to/educhat_model \
--top_k 50 \
--do_sample True \
--max_new_tokens 512
```

启动demo后，你可以将链接分享给朋友，通过网页与EduChat交互

#### Api Demo

你可以运行仓库中的[demo/educhat_api.py](https://github.com/icalk-nlp/EduChat/blob/main/demo/educhat_api.py)来对外提供一个简单的api服务

```bash
python educhat_api.py
```

启动api服务后，你可以通过网络调用来与EduChat交互

```bash
## curl EduChat
curl -X POST "http://localhost:19324" \
     -H 'Content-Type: application/json' \
     -d '{"prompt": "你是谁？"}'
```

首次调用，你会得到一个api服务返回的uid

```json
{"response":"\n<|Worm|>: 你好，有什么我可以帮助你的吗？","history":[["你好","\n<|Worm|>: 你好，有什么我可以帮助你的吗？"]],"status":200,"time":"2023-04-28 09:43:41","uid":"10973cfc-85d4-4b7b-a56a-238f98689d47"}
```

你可以在后续的对话中填入该uid来和EduChat进行多轮对话

```bash
## curl EduChat multi-round
curl -X POST "http://localhost:19324" \
     -H 'Content-Type: application/json' \
     -d '{"prompt": "你是谁？", "uid":"10973cfc-85d4-4b7b-a56a-238f98689d47"}'
```


## EduChat系列产品

- **产品1**: 舒心阁（MindCare@EduChat）
  学生的心理健康已成为当下教育中不可忽视的问题，也隐性影响着学习效果。团队以EduChat为核心基座，研发了涵盖心理评估、浅层情感陪伴、深层心理疏导的“舒心阁”（MindCare@EduChat）。
  - 心理评估：以专业量表为基础，通过主动问询式对话交互，并结合语音、面部表情等多模态情绪识别，实现精准、高效的心理健康指数评估，赋能大规模人群的心理问题快速筛查和早期研判。
  - 浅层情感陪伴：定制青少年喜爱的角色，并结合探索、安抚、共情等通用策略，提供特定角色的情感支持和陪伴。
  - 深层心理疏导：针对当前大模型过早给出通用建议、过度共情（“彩虹屁”），难以深度溯因和真正帮助用户走出心理困境的问题，团队融合心理学中常用的情绪聚焦疗法（EFT）、认知行为疗法（CBT）等理论，研发了具备疗愈特性的深层心理疏导技术，引导大语言模型模拟咨询师进行心理困境分析、心理疏导目标拆分、分阶段计划任务设置、行动策略选择等，全面守护青少年心理健康。
- **产品2**: 安全加固（Shell@EduChat）
- **产品3**: 奇迹疗愈（MiracleH@EduChat）
- **产品4**: AI智慧黑板（AiBoard@EduChat）
- **产品**5: 敏捷出版（AgiPub@EduChat）


## :construction: 未来计划

从EduChat-001到EduChat-002的迭代过程中，我们逐步增强了它的中文能力、忠实度、安全度和有帮助性方面的表现。然而，EduChat-002仍然是一个早期模型，我们的旅程也才刚刚开始。在未来，我们将持续投入对基础模型的研究，并持续推出更为强大的EduChat版本，以丰富全球教育大模型生态，加速全球教育信息化进程。

- **逻辑推理**：逻辑推理能力是衡量大模型性能的重要指标，我们计划通过增大语言模型基座、增强特定训练数据等手段强化EduChat的逻辑推理能力；
- **个性化辅导**：我们期望的EduChat应当是千人千面的，未来我们希望能够给每个人一个独一无二的EduChat，它将在与你的交互中持续学习，伴随你的成长而成长，成为你的专属助手。
- **工具调用**：语言模型本身具有明显的局限性，例如符号运算能力弱，我们计划在后续升级EduChat，使其具备调用外部工具能力，帮助其更好地进行生成。


## :page_with_curl: 开源协议、模型局限、使用限制与免责声明

本项目所含代码采用[Apache 2.0](https://github.com/icalk-nlp/EduChat/blob/main/LICENSE)协议，数据采用[CC BY-NC 4.0](https://github.com/icalk-nlp/EduChat/blob/main/DATA_LICENSE)协议。

尽管我们对EduChat进行了优化，但仍存在以下问题，需要进行改进：

- 当涉及到事实性指令时，可能会产生错误的回答，与实际事实相悖。

- 模型回复可能存在偏见，有可能生成危险性言论。

- 在某些场景中，比如推理、代码、多轮对话等方面，模型的能力仍有待提高。

鉴于上述模型的局限性，我们要求开发者仅将我们开源的代码、数据、模型以及由该项目生成的衍生物用于研究目的，禁止用于商业用途，以及其他可能对社会带来危害的用途。

本项目仅供研究目的使用，项目开发者对于使用本项目（包括但不限于数据、模型、代码等）所导致的任何危害或损失不承担责任。详情请参考该[免责声明](https://github.com/icalk-nlp/EduChat/blob/main/LICENSE/DISCLAIMER)。

## 团队介绍
- **主要发起人**: 陈琴、周杰、贺樑
- **主要负责人**: 陈琴、周杰、吴雯、吴兴蛟、吴玉兰、贺樑
- **参与人**: 丁宇洋、但宇豪、周友根、王子威、李俊松、丁棋、周莘杰、宋知时、杨宇涛、怀天宇、詹必豪、沈锴成、单良、许俊杰、张子昊、贝佳洋

## :heart: 致谢

- [Qwen](https://github.com/QwenLM/Qwen)，[Baichuan](https://github.com/baichuan-inc): EduChat是基于Qwen和Baichuan作为基座
- [Open Assistant](https://github.com/LAION-AI/Open-Assistant): EduChat参考OA构建模型训练代码
- [华东师范大学出版社](https://www.ecnupress.com.cn/)：[教育大模型语料](http://educorpus.ecnupress.com.cn/#/)由华师大出版社支持
- [竹蜻蜓数据科技（浙江）有限公司](https://www.autopaddle.com//): 开发支持
- [邱锡鹏教授](https://xpqiu.github.io/): 项目顾问
