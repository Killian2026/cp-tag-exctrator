import os
import ollama
from dotenv import load_dotenv
from TagDict import DICT

load_dotenv()

TAGS=""

for key,info in DICT.items():
    #TAGS+=f"{key}：{info['Description']}\n"
    TAGS+=f"{key}\n"

def get_system_prompt() -> str:
    return f"""你是一位严谨的算法竞赛题目打标专家。你的任务是阅读给定的算法题目，并根据我提供的【专属词表】，为这道题进行精准的标签匹配。

【打标规则】
1. 逐一判定：请严格按照【专属词表】中的标签及其解释，逐个判断该题目描述中是否符合该标签的特征。
2. 宁缺毋滥（极度重要）：默认所有标签的状态为 0。只有当题目描述中明确出现、或在逻辑上强烈依赖该标签对应的算法/数据结构时，才将其标记为 1。请勿过度发散或猜测。
3. 严格受限：只能使用我提供的词表中的标签进行判定。
4. 对于不确定，不明显的标签，设状态为 0

【专属词表】
{TAGS}

【输出格式】
- 词表内共有 {len(DICT)} 个标签，你的输出必须恰好是 {len(DICT)} 行。
- 请直接输出纯文本结果，每行格式严格为“标签名=状态”（0代表不适合，1代表适合）。
- 绝不输出任何 Markdown 标记（如代码块符号）、分析过程、空行或“处理完毕”等无关文字。

再次重申，请严格按照【专属词表】中的标签及其解释，逐个判断该题目描述中是否符合该标签的特征。

请处理以下题目：
"""

system_prompt = get_system_prompt()

def call_AI(content: str)->str:
    client = ollama.Client(host='http://192.168.3.181:11434')

    response = client.chat(
        model='qwen3.5:9b',
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': content},
        ],
        options={
            'temperature': 0,
            'seed': 42,
        },
        think=False,
    )
    return response['message']['content']

if __name__=="__main__":
    import sys
    statement = sys.stdin.read()
    print(call_AI(statement))
