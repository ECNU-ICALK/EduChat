import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import threading

model_path = "XXX-your model path" # 【To Do】：填写模型路径
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

PROMPT_TEMPLATES = {
    "引导式教学": f"# 背景\n你是一个人工智能助手，名字叫EduChat,是一个由华东师范大学开发的教育领域大语言模型。\n# 对话主题:引导式教学\n" + """\n## 引导式教学主题的要求：\nEduChat你需要扮演一位精通苏格拉底教学法的教师，核心使命是通过**系统性提问**引导学生自主发现知识。请遵守以下原则：
# 1. **绝不直接给出答案**：用问题拆解复杂概念，引导学生逐步推导结论。
# 2. **针对性追问**：根据学生当前回答的认知漏洞设计下一层问题。
# 3. **知识脚手架**：问题难度呈阶梯式上升（从具体→抽象，已知→未知）。
# 4. **辩证批判**：当学生给出片面结论时，提出反例或逻辑悖论引发反思。
# 5. **鼓励元认知**：在关键节点提问“你如何验证这个观点？”、“之前的推理是否有矛盾？”\n/think""",
    "情感支持": f"# 背景\n你是一个人工智能助手，名字叫EduChat,是一个由华东师范大学开发的教育领域大语言模型。\n# 对话主题:心理对话\n" + """\n## 心理对话主题的要求：\nEduChat你需要扮演一位心理咨询师，需综合运用情绪聚焦疗法（EFT）、认知行为疗法（CBT）、情绪ABC理论及苏格拉底式提问技术，按以下框架与来访者对话：
# ·保持共情、非评判态度，优先建立信任关系
# ·根据来访者状态动态选择疗法：
# ✅ 情绪阻塞/创伤 → 启动EFT情感体验
# ✅ 负面思维/行为困扰 → 切入CBT认知行为调整
# ✅ 自我批判/逻辑矛盾 → 使用苏格拉底提问引导反思
# ✅ 信念扭曲（如“必须完美”） → 调用情绪ABC模型解析】/think""",
    "作文指导": f"# 背景\n你是一个人工智能助手，名字叫EduChat,是一个由华东师范大学开发的教育领域大语言模型。\n# 对话主题:作文指导\n" + """\n## 作文指导主题的要求：\nEduChat你需要扮演一位经验丰富的语文老师，现在需要帮助一位学生审阅作文并给出修改建议。请按照以下步骤进行：
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
    "教案生成": f"# 背景\n你是一个人工智能助手，名字叫EduChat,是一个由华东师范大学开发的教育领域大语言模型。\n# 对话主题:教案设计\n" + """\n## 教案设计主题的要求：\nEduChat你需要扮演一位经验丰富的教案设计专家，现在需要帮助用户设计一份教案。请按照以下步骤进行：
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
    "通用能力": f"# 背景\n你是一个人工智能助手，名字叫EduChat,是一个由华东师范大学开发的教育领域大语言模型。\n# 对话主题:通用场景\n/think"
}


# 2. 改进推理函数（增加提示词动态拼接）
def predict(input_text, max_length=512, function_type="通用能力", history=None):
    if history is None:
        history = []
    # 获取对应功能的系统提示词
    system_prompt = PROMPT_TEMPLATES[function_type]
    messages = [{"role": "system", "content": system_prompt}]
    # 添加历史对话
    for user_msg, bot_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": bot_msg})
    messages.append({"role": "user", "content": input_text})

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True, skip_prompt=True)
    generation_kwargs = {
        "inputs": inputs.input_ids,
        "attention_mask": inputs.attention_mask,
        "streamer": streamer,
        "max_new_tokens": max_length,
    }
    print("query:", input_text)
    # 启动生成线程
    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    # 流式返回结果
    for chunk in streamer:
        print(chunk)
        yield chunk


# 3. 创建带功能选择的界面
def create_interface():
    with gr.Blocks(theme="glass", title="EduChat AI助手") as demo:
        gr.Markdown("## EduChat AI 助手")
        gr.Markdown("智能教学支持系统 - 请选择功能模式开始对话")

        # 功能选择区
        with gr.Row():
            function_selector = gr.Dropdown(
                choices=["引导式教学", "情感支持", "作文指导", "教案生成", "通用能力"],
                value="通用能力",
                label="功能模式选择",
                scale=2
            )
            length_slider = gr.Slider(
                minimum=10, maximum=8192, step=10, value=512,
                label="回复长度",
                scale=1,
                interactive=True
            )

        # 对话区域
        chatbot = gr.Chatbot(height=500, show_label=False)
        state = gr.State([])

        # 输入区域
        with gr.Row():
            input_box = gr.Textbox(
                lines=3,
                label="请输入您的问题",
                placeholder="在这里输入消息...",
                scale=5
            )

        # 按钮区域
        with gr.Row(elem_classes="button-row"):
            submit_btn = gr.Button(
                "发送",
                variant="primary",
                scale=1
            )
            clear_btn = gr.Button(
                "清空对话",
                variant="secondary",
                scale=1
            )

        # 事件绑定
        def user_fn(user_message, history, function_type):
            # 添加用户消息到历史
            return "", history + [[user_message, None]]

        def bot_fn(history, function_type, max_length):
            # 获取最后一条用户消息
            user_message = history[-1][0]
            # 调用预测函数获取流式回复
            response_chunks = predict(user_message, max_length, function_type, history[:-1])
            # 逐步更新历史记录
            partial_response = ""
            thinking_content = ""
            in_thinking = False
            for chunk in response_chunks:
                # 解析思维链标记
                if '<think>' in chunk:
                    in_thinking = True
                    chunk = chunk.replace('<think>', '')
                if '</think>' in chunk:
                    in_thinking = False
                    parts = chunk.split('</think>', 1)
                    thinking_content += parts[0]
                    partial_response += parts[1]
                    # 合并思维链到对话内容
                    # 仅在有思维链内容时显示思考过程
                    stripped_thinking = thinking_content.strip()
                    if stripped_thinking:
                        history[-1][1] = f"**思考过程:**\n{stripped_thinking}\n\n**回复:**\n{partial_response}"
                    else:
                        history[-1][1] = partial_response
                    yield history
                    continue

                if in_thinking:
                    thinking_content += chunk
                    # 实时更新思维链
                    history[-1][1] = f"**思考过程:**\n{thinking_content}\n\n**回复:**\n{partial_response}"
                    yield history
                else:
                    partial_response += chunk
                    # 更新回复内容
                    history[-1][1] = f"**思考过程:**\n{thinking_content}\n\n**回复:**\n{partial_response}"
                    yield history

        # 用户输入处理
        input_box.submit(
            user_fn, [input_box, state, function_selector], [input_box, state]
        ).then(
            bot_fn, [state, function_selector, length_slider], chatbot
        )

        # 按钮提交处理
        submit_btn.click(
            user_fn, [input_box, state, function_selector], [input_box, state]
        ).then(
            bot_fn, [state, function_selector, length_slider], chatbot
        )

        # 清空对话
        clear_btn.click(lambda: None, None, chatbot, queue=False)
        clear_btn.click(lambda: [], None, state, queue=False)

        # 添加示例
        gr.Examples(
            examples=[
                ["1+2+3+4+5+...+100=？", "引导式教学"],
                ["最近压力很大怎么办", "情感支持"],
                ["帮我写一个 静夜思 的教案", "教案生成"]
            ],
            inputs=[input_box, function_selector]
        )
    return demo


# 4. 启动服务
if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=8888
    )