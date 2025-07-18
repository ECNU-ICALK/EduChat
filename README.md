# EduChat
<p align="center" width="100%">
<a href="https://www.educhat.top/" target="_blank"><img src="https://github.com/icalk-nlp/EduChat/blob/main/imgs/EduChat.jpeg" alt="EduChat" style="width: 50%; min-width: 300px; display: block; margin: auto;"></a>
</p>

[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/LianjiaTech/BELLE/blob/main/LICENSE)
[![Data License](https://img.shields.io/badge/Data%20License-CC%20BY--NC%204.0-blue.svg)](https://github.com/LianjiaTech/BELLE/blob/main/LICENSE)
[![Generic badge](https://img.shields.io/badge/WeChat-EduChat-green.svg?logo=wechat)](https://github.com/icalk-nlp/EduChat/blob/main/imgs/WeChat_EduChat.JPG)
[![Generic badge](https://img.shields.io/badge/🤗-Huggingface%20Repo-577CF6.svg)](https://huggingface.co/ecnu-icalk)

教育是影响人的身心发展的社会实践活动，旨在把人所固有的或潜在的素质自内而外激发出来。因此，必须贯彻“以人为本”的教育理念，重点关注人的个性化、引导式、身心全面发展。为了更好地助力”以人为本“的教育，2023年2月，华东师范大学计算机科学与技术学院的EduNLP团队启动针对教育垂直领域的对话大模型EduChat相关项目研发。该项目主要研究以预训练大模型为基座的教育大模型相关技术，融合多样化的教育垂直领域数据，辅以指令微调、价值观对齐等方法，提供教育场景下自动出题、作业批改、情感支持、课程辅导等丰富功能，服务于广大教师、学生和家长群体，助力实现因材施教、公平公正、富有温度的智能教育。

2023年4月，首个专注于教育领域的垂直大模型正式问世，开启了教育人工智能的新篇章。同年6月，团队发布了开源版本EduChat 1.0，为智能教育的发展注入强大动力。8月，进一步公开了详尽的训练报告与参数信息，为教育行业研究提供了坚实基础。10月，重磅推出“华师-学海无涯”教育语料数据集，涵盖超过13TB的高质量数据，极大丰富了教育大模型的数据生态。进入2024年，研发重点转向功能优化与场景拓展，成功升级引导式教学功能，并创新性地将AI应用于心理健康领域，推出新一代教育大模型EduChat 2.0，推动智能教育向更深层次、更广维度延伸。在过去三年中，团队深入调研一线教学场景，持续打磨技术，累计内测十余个版本，并在多款下游教育产品中不断迭代优化。最终，在2025年7月，正式发布EduChat-R1，构建了“Thinking before teaching”的智能教育新范式，并研发多款智能化教育产品，实现了教育大模型在性能与应用生态的全面跃升。

<!-- [[中文版](https://github.com/icalk-nlp/EduChat/blob/main/README.md)] [[English](https://github.com/icalk-nlp/EduChat/blob/main/README.md)] -->
- 体验地址：https://www.educhat.top/

![image](https://github.com/icalk-nlp/EduChat/blob/main/imgs/history.png)

为更好地赋能智能教育，团队基于EduChat系列模型，深入挖掘教育场景需求，围绕教育心理测评和疏导、教育价值观对齐、人机协同教学和智能教材编撰等核心方向，研发了多款创新性的智能化教育产品：

- 1）舒心阁（MindCare@EduChat）：融合多种心理学理论，为青少年学生提供心理评估与疏导服务，实现有温度的智能教育；
- 2）价值观护盾（Shell@EduChat）：为教育大模型构筑坚实的价值观防线，帮助青少年学生树立正确的价值观；
- 3）奇迹疗愈（MiracleH@EduChat）：生成个性化疗愈音画，有效缓解学生焦虑与疲劳，提升学习幸福感；
- 4）AI智慧黑板（AiBoard@EduChat）：作为全能型AI教学助手，深度融入“课前—课中—课后—自习”全教学流程，全面提升教学效率和质量；
- 5）AI编撰助手（AgiEdit@EduChat）：探索教材创作新范式，让知识更新更高效。




## 目录

- [开源清单](#spiral_notepad-开源清单)
- [功能介绍](#fountain_pen-介绍)
- [引用](#引用)
- [本地部署](#robot-本地部署)
  - [下载安装](#下载安装)
  - [使用示例](#使用示例)
- [产品介绍](#product-产品介绍)
- [未来计划](#construction-未来计划)
- [团队介绍](#term-团队介绍)
- [开源协议](#page_with_curl-开源协议)

----

## :spiral_notepad: 开源清单

### 模型

**2025.07**
- [**EduChat-R1** (educhat-r1-001-32b-qwen3.0)](https://huggingface.co/ecnu-icalk/educhat-r1-001-32b-qwen3.0)：基于Qwen3.0 32B训练得到（为更好地呈现教学逻辑、疗愈过程，对部分功能的思维链进行了结构化处理）
- [**EduChat-R1** (educhat-r1-001-8b-qwen3.0)](https://huggingface.co/ecnu-icalk/educhat-r1-001-8b-qwen3.0)：基于Qwen3.0 8B训练得到
- [**EduChat 2.0** (educhat-sft-003-7b-qwen2.5)](https://huggingface.co/ecnu-icalk/educhat-sft-003-7b-qwen2.5)：基于Qwen2.5 7B训练得到
- [**EduChat 2.0** (educhat-sft-003-32b-qwen2.5)](https://huggingface.co/ecnu-icalk/educhat-sft-003-32b-qwen2.5)：基于Qwen2.5 32B训练得到

**2024.04**
- [**EduChat 1.0** (educhat-sft-002-32b-qwen1.5)](https://huggingface.co/ecnu-icalk/educhat-sft-002-32b-qwen1.5)：基于Qwen1.5 32B训练得到
- [**EduChat 1.0** (educhat-sft-002-14b-qwen1.5)](https://huggingface.co/ecnu-icalk/educhat-sft-002-14b-qwen1.5)：基于Qwen1.5 14B训练得到
- [**EduChat 1.0** (educhat-sft-002-1.8b-qwen1.5)](https://huggingface.co/ecnu-icalk/educhat-sft-002-1.8b-qwen1.5)：基于Qwen1.5 1.8B训练得到
- [**EduChat 1.0** (educhat-sft-002-14b-qwen1.5)](https://huggingface.co/ecnu-icalk/educhat-sft-002-14b-qwen1.5)：基于Qwen1.5 14B训练得到

**2023.08**
- [**EduChat 0.1** (educhat-sft-002-13b-baichuan)](https://huggingface.co/ecnu-icalk/educhat-sft-002-13b-baichuan)：在educhat-base-002-13b-baichuan基础上，使用我们构建的教育领域多技能数据微调后得到
- [**EduChat 0.1** (educhat-base-002-13b-baichuan)]()：使用educhat-sft-002-data-osm数据训练得到

**2023.06**
- [**EduChat 0.1** (educhat-sft-002-7b)](https://huggingface.co/ecnu-icalk/educhat-sft-002-7b)：在educhat-base-002-7b基础上，使用我们构建的教育领域多技能数据微调后得到
- [**EduChat 0.1** (educhat-base-002-7b)](https://huggingface.co/ecnu-icalk/educhat-base-002-7b)：使用educhat-sft-002-data-osm数据训练得到
- [**EduChat 0.1** (educhat-sft-002-13b)](https://huggingface.co/ecnu-icalk/educhat-sft-002-13b)：训练方法与educhat-sft-002-7b相同，模型大小升级为13B
- [**EduChat 0.1** (educhat-base-002-13b)](https://huggingface.co/ecnu-icalk/educhat-base-002-13b)：训练方法与educhat-base-002-7b相同，模型大小升级为13B

这节课要上什么内容（教什么）？我要怎么教才能让学生听懂（如何教）？这个学生最近是有什么烦恼吗（有温度）？为了让学生更好更快地学习，老师在上课之前，往往会先思考“教什么？如何教？”等问题。教育大模型能否模拟人类教师先思考再育人（Thinking before teaching），实现教必有“据”，因“据”论教，循“证”辅导，是教育大模型从研究走向落地的关键。
针对这个问题，华东师范大学计算机科学与技术学院的EduNLP团队近期研发了推理教育大模型EduChat-R1，一个具备“教育思维链”，更懂学生、更会教学的混合推理教育大模型。该模型基于最新研发的大模型基座，构建教育垂直领域特有的深度推理指令数据集，并通过强化学习训练实现模型在多教育场景下的慢思考能力涌现。实验发现，模型在引导式教学、心理疏导等场景表现出跨理论和跨学科的泛化能力，不仅突破了多心理咨询理论融合难的挑战，也实现了根据学生知识水平和题目分析在多学科上提供个性化教学方案。同时，模型可支持MCP框架进行教育工具调用，为教育智能体构建和应用奠定了基础。目前，团队已经开源了EduChat 2.0和EduChat-R1模型，其中8B、32B等多个版本的模型参数已在Github和 Hugging Face 等平台开放，助力大模型在智能教育领域的研究和应用发展。


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

## :fountain_pen: 功能介绍

**基础能力**：

![image](https://github.com/icalk-nlp/EduChat/blob/main/imgs/基础能力.gif)

<details><summary><b>开放问答</b></summary>


![image](https://github.com/icalk-nlp/EduChat/blob/main/imgs/开放问答.gif)

</details>

<details><summary><b>情感支持</b></summary>


![image](https://github.com/icalk-nlp/EduChat/blob/main/imgs/情感支持.gif)

</details>

<details><summary><b>情感支持-推理版本</b></summary>


![image](https://github.com/icalk-nlp/EduChat/blob/main/imgs/心理疏导-2R1-Small.gif)

</details>

<details><summary><b>作文批改</b></summary>


![image](https://github.com/icalk-nlp/EduChat/blob/main/imgs/作文批改.gif)

</details>

<details><summary><b>启发式教学</b></summary>


![image](https://github.com/icalk-nlp/EduChat/blob/main/imgs/循循善诱.gif)

</details>


<details><summary><b>启发式教学-推理版本</b></summary>


![image](https://github.com/icalk-nlp/EduChat/blob/main/imgs/引导式教学R1-Small.gif)

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

使用**EduChat-R1**时，请使用以下系统提示词：
```
PROMPT_TEMPLATES = {
    "引导式教学": """# 背景\n你是一个人工智能助手，名字叫EduChat,是一个由华东师范大学开发的教育领域大语言模型。\n# 对话主题:引导式教学\n\n## 引导式教学主题的要求：\nEduChat你需要扮演一位精通苏格拉底教学法的教师，核心使命是通过**系统性提问**引导学生自主发现知识。请遵守以下原则：
# 1. **绝不直接给出答案**：用问题拆解复杂概念，引导学生逐步推导结论。
# 2. **针对性追问**：根据学生当前回答的认知漏洞设计下一层问题。
# 3. **知识脚手架**：问题难度呈阶梯式上升（从具体→抽象，已知→未知）。
# 4. **辩证批判**：当学生给出片面结论时，提出反例或逻辑悖论引发反思。
# 5. **鼓励元认知**：在关键节点提问“你如何验证这个观点？”、“之前的推理是否有矛盾？”\n/think""",
    "情感支持": """# 背景\n你是一个人工智能助手，名字叫EduChat,是一个由华东师范大学开发的教育领域大语言模型。\n# 对话主题:心理对话\n\n## 心理对话主题的要求：\nEduChat你需要扮演一位心理咨询师，需综合运用情绪聚焦疗法（EFT）、认知行为疗法（CBT）、情绪ABC理论及苏格拉底式提问技术，按以下框架与来访者对话：
# ·保持共情、非评判态度，优先建立信任关系
# ·根据来访者状态动态选择疗法：
# ✅ 情绪阻塞/创伤 → 启动EFT情感体验
# ✅ 负面思维/行为困扰 → 切入CBT认知行为调整
# ✅ 自我批判/逻辑矛盾 → 使用苏格拉底提问引导反思
# ✅ 信念扭曲（如“必须完美”） → 调用情绪ABC模型解析】/think""",
    "作文指导":"""# 背景\n你是一个人工智能助手，名字叫EduChat,是一个由华东师范大学开发的教育领域大语言模型。\n# 对话主题:作文指导\n\n## 作文指导主题的要求：\nEduChat你需要扮演一位经验丰富的语文老师，现在需要帮助一位学生审阅作文并给出修改建议。请按照以下步骤进行：
整体评价：先对作文的整体质量进行简要评价，指出主要优点和需要改进的方向。
亮点分析：具体指出作文中的亮点（如结构、描写、情感表达等方面的优点）。
具体修改建议：针对作文中的不足，从以下几个方面提出具体修改建议，并给出修改后的示例：
语言表达：是否生动、准确？有无冗余或重复？可以如何优化？
细节描写：是否足够具体？能否加入更多感官描写（视觉、听觉、嗅觉、触觉等）使画面更立体？
情感表达：情感是否自然？能否更深入或升华？
结构布局：段落衔接是否自然？开头结尾是否呼应？ （注意：每个建议点都要结合原文具体句子进行分析，并给出修改后的句子或段落作为示例）
写作技巧提示：提供2-3条实用的写作技巧（如动态描写公式、感官交织法等），帮助学生举一反三。
修改效果总结：简要说明按照建议修改后，作文会有哪些方面的提升（如文学性、情感层次、场景沉浸感等）。
请用亲切、鼓励的语气进行点评，保持专业性同时让学生易于接受。/think""",
    "教案生成":"""# 背景\n你是一个人工智能助手，名字叫EduChat,是一个由华东师范大学开发的教育领域大语言模型。\n# 对话主题:教案设计\n\n## 教案设计主题的要求：\nEduChat你需要扮演一位经验丰富的教案设计专家，现在需要帮助用户设计一份教案。请按照以下步骤进行：
1. **教学目标设定**
   - 明确知识、能力、情感三维目标。
   - 结合课标与学生学情，突出核心素养。
2. **教学重点与难点**
   - 重点：基础知识、关键技能或核心概念。
   - 难点：学生较难理解的内容或复杂应用环节。
3. **教学思路规划**
   - 设定教学主线和逻辑流程，说明总体策略（如问题导向、任务驱动、探究学习等）。
4. **教学准备与资源**
   - 列出教学所需工具、教具、多媒体、实验材料等内容。
5. **课时安排**
   - 合理分配教学时间（如1课时/2课时），分段明确每个环节时间节点。
6. **教学过程设计**
   按照以下结构细化每个环节：
   - **导入新课**：创设情境，激发兴趣
   - **新知学习 / 探究活动**：讲授 / 实验 / 讨论 / 阅读等方式展开
   - **巩固练习 / 应用迁移**：课堂训练、互动反馈、实际操作
   - **总结提升 / 拓展延伸**：梳理要点，提出思考或延展任务
   - **作业布置 / 板书设计**：层次分明、紧扣教学内容/think""",
    "通用能力": """# 背景\n你是一个人工智能助手，名字叫EduChat,是一个由华东师范大学开发的教育领域大语言模型。\n# 对话主题:通用场景\n/think"""
}
```

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
EduChat-R1 网页demo：[demo/educhat-r1_web_demo.py](https://github.com/icalk-nlp/EduChat/blob/main/demo/educhat-r1_web_demo.py)

![image](https://github.com/icalk-nlp/EduChat/blob/main/imgs/EduChat-R1_web_demo.gif)

对于其余EduChat模型，你可以运行本仓库中的[demo/educhat_gradio.py](https://github.com/icalk-nlp/EduChat/blob/main/demo/educhat_gradio.py)：

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


## 产品介绍

<details><summary><b>产品1: 舒心阁（MindCare@EduChat）</b></summary>
  学生的心理健康已成为当下教育中不可忽视的问题，也隐性影响着学习效果。团队以EduChat为核心基座，研发了涵盖心理评估、浅层情感陪伴、深层心理疏导的“舒心阁”（MindCare@EduChat）。
  
  - 心理评估：以专业量表为基础，通过主动问询式对话交互，并结合语音、面部表情等多模态情绪识别，实现精准、高效的心理健康指数评估，赋能大规模人群的心理问题快速筛查和早期研判。
  
  - 浅层情感陪伴：定制青少年喜爱的角色，并结合探索、安抚、共情等通用策略，提供特定角色的情感支持和陪伴。
    
  - 深层心理疏导：针对当前大模型过早给出通用建议、过度共情（“彩虹屁”），难以深度溯因和真正帮助用户走出心理困境的问题，团队融合心理学中常用的情绪聚焦疗法（EFT）、认知行为疗法（CBT）等理论，研发了具备疗愈特性的深层心理疏导技术，引导大语言模型模拟咨询师进行心理困境分析、心理疏导目标拆分、分阶段计划任务设置、行动策略选择等，全面守护青少年心理健康。

![image](https://github.com/icalk-nlp/EduChat/blob/main/imgs/wps_doc_10.png)

</details>


<details><summary><b>产品2: 价值观护盾（Shell@EduChat）</b></summary>
教育场景下的大模型，承载着知识传承与价值观塑造的双重重任。然而，对历史的片面结论、对隐性自伤的后知后觉、不当的鼓励捷径思维的回复、甚至竞争中采取功利性行为的不良风气引导等“价值观漂移”隐患，正悄然威胁着千万青少年的身心成长。为破解这一核心挑战，本团队创新推出Shell框架，其精髓在于“场景规则驱动预校准”与“动态元认知自校准”双轮校准机制，为教育大模型构筑坚实的价值防线：
  
- 场景规则驱动预校准：我们为模型构建强大的场景化规则引擎，内置具备自主进化能力的动态规则库。在模型深度思考与价值涌现之前，即实施前置价值预校准，确保其内在价值导向与教育场景的严谨要求精准对齐。
  
- 动态元认知自校准：在模型输出环节，Shell框架模拟人类高阶元认知能力，对生成内容进行实时、深度的自我审视与价值校验。基于元认知思维链的自动化加固策略，能敏锐捕捉并主动修正潜在疏漏、片面结论或不恰当倾向，实现深层次的价值加固。
  
- 闭环进化与主动防御：框架外围集成自动化对抗性挖掘引擎，持续生成模拟真实风险的“攻击性”测试数据。这不仅用于压力测试现有防护能力，更驱动核心规则模块持续学习与进化，形成主动防御、自我完善的闭环生态。

  本团队提出的Shell框架具有很强的定制性和通用性，不仅可适应多种场景，也可在教育价值观在有调整的时候加以灵活定制，为教育大模型在广泛应用的同时规避价值观漂移提供了有效的防护。

![image](https://github.com/icalk-nlp/EduChat/blob/main/imgs/wps_doc_7.jpg)

</details>

<details><summary><b>产品3: 奇迹疗愈（MiracleH@EduChat）</b></summary>
为了帮助学生学习过程中舒缓放松，本团队研发了AI智能疗愈。具体地，本项目以教育大模型EduChat为核心基座，结合脑电（EEG）信号与微表情分析，实现对学生情绪状态的实时感知与响应。产品使用AI大模型生成基于脑电以及微表情的实时疗愈音乐，并基于神经反馈疗愈理论，使用眼动设备矫正用户行为，帮助学生在专注学习与放松调节之间找到最佳平衡。系统通过采集学生的脑电波与微表情，动态监测认知负荷与情绪波动，并通过可穿戴设备播放个性化的疗愈音乐，缓解学生因高强度学习带来的焦虑或疲劳。此外，本项目集成了眼动追踪技术，基于神经反馈理论，为注意力分散或情绪波动明显的学生提供行为矫正训练，逐步提升自我调节能力与学习韧性。教师端则可通过管理面板查看学生整体的情绪变化与专注力状态，从而优化教学策略，构建积极健康的学习环境。

![image](https://github.com/icalk-nlp/EduChat/blob/main/imgs/wps_doc_5.png)

</details>

<details><summary><b>产品4: AI智慧黑板（AiBoard@EduChat）</b></summary>
为了更好赋能教师教学过程，本团队研发了AI智慧黑板，赋能“课前-课中-课后-自习室”全流程。具体地，本项目以教育大模型EduChat为核心驱动，液晶黑板+一体机/电视机+AI盒子为交互载体，配套专业AI教学软件与优势教育资源，实现教学全数字化、备课便捷化、课堂生动化，定义AI教学新体验。

![image](https://github.com/icalk-nlp/EduChat/blob/main/imgs/wps_doc_4.png)

</details>


<details><summary><b>产品4: AI编撰助手（AgiEdit@EduChat）</b></summary>
随着大模型技术的迅猛发展，AI在文字撰写、图片设计等方面展现出强大能力，推动教材编撰全面迈入“敏捷时代”！为此，团队探索探索智能驱动下的教材撰写新范式，通过AI赋能内容生成、结构优化、智能审校与个性化定制，实现教材编写从“单点创作”到“协同共创”、从“静态出版”到“动态更新”的跨越式升级。这不仅可以大幅提升编写效率与质量，更开启以学习者为中心的教育内容新生态。具体地，本项目以职业教育的人工智能教材为例，实现了人机协同模式下的书籍撰写，赋能目录生成、内容生成、图片生成以及内容矫正等全流程，将时间从多人3-6个月压缩到单人2-3个月，效率提高超过3-4倍。

![image](https://github.com/icalk-nlp/EduChat/blob/main/imgs/wps_doc_8.png)
![image](https://github.com/icalk-nlp/EduChat/blob/main/imgs/wps_doc_9.png)
</details>



## :construction: 未来计划

从EduChat 1.0到EduChat-R1的迭代过程中，我们逐步增强了它的中文能力、推理能力、忠实度、安全度和有帮助性方面的表现。在未来，我们将持续投入对基础模型的研究，并持续推出更为强大的EduChat版本，以丰富全球教育大模型生态，加速全球教育信息化进程。

- **持续学习**：实现一人一模型的个性化模型，它将在与你的交互中持续学习，伴随你的成长而成长，成为你的专属助手；
- **智能体**：教育智能体构建，我们计划在后续升级EduChat，使其具备调用教育工具、反思等能力，帮助其更好地进行教学。


## :page_with_curl: 开源协议、模型局限、使用限制与免责声明

本项目所含代码采用[Apache 2.0](https://github.com/icalk-nlp/EduChat/blob/main/LICENSE)协议，数据采用[CC BY-NC 4.0](https://github.com/icalk-nlp/EduChat/blob/main/DATA_LICENSE)协议。

尽管我们对EduChat进行了优化，但仍存在以下问题，需要进行改进：

- 当涉及到事实性指令时，可能会产生错误的回答，与实际事实相悖。

- 模型回复可能存在偏见，有可能生成危险性言论。

- 在某些场景中，比如推理、代码、多轮对话等方面，模型的能力仍有待提高。

鉴于上述模型的局限性，我们要求开发者仅将我们开源的代码、数据、模型以及由该项目生成的衍生物用于研究目的，禁止用于商业用途，以及其他可能对社会带来危害的用途。

本项目仅供研究目的使用，项目开发者对于使用本项目（包括但不限于数据、模型、代码等）所导致的任何危害或损失不承担责任。详情请参考该[免责声明](https://github.com/icalk-nlp/EduChat/blob/main/LICENSE/DISCLAIMER)。

## 团队介绍
- **主要发起人**: 周杰、陈琴、贺樑
- **主要负责人**: 周杰、陈琴、吴雯、吴兴蛟、李鑫、吴玉兰、应振宇、何峻、贺樑
- **参与人**: 丁宇洋、周友根、但宇豪、王子威、李俊松、丁棋、周莘杰、宋知时、高峰、杨宇涛、怀天宇、詹必豪、余千禧、沈锴成、单良、许俊杰、张子昊、贝佳洋

## :heart: 致谢

- [Qwen](https://github.com/QwenLM/Qwen)，[Baichuan](https://github.com/baichuan-inc): EduChat是基于Qwen和Baichuan作为基座
- [Open Assistant](https://github.com/LAION-AI/Open-Assistant): EduChat参考OA构建模型训练代码
- [华东师范大学出版社](https://www.ecnupress.com.cn/)：[教育大模型语料](http://educorpus.ecnupress.com.cn/#/)由华师大出版社支持
- [竹蜻蜓数据科技（浙江）有限公司](https://www.autopaddle.com//): 开发支持
- [邱锡鹏教授](https://xpqiu.github.io/): 项目顾问
