"""
算法题重题检测 —— 推理入口，使用训练好的嵌入模型做相似题检索。

  题目文本 -> LLM打标 -> 多热向量 -> 嵌入模型 -> 32维嵌入 -> 与库中所有题的嵌入做余弦相似度

用法:
    python main.py < problem.txt        # 从文件读入
    echo "N cows..." | python main.py   # 管道输入
"""

import sys
import torch
import tool
from model import TagEmbedding
from dataset import tags_to_multihot, NUM_TAGS
from math import sqrt


# ═══════════════════════════════════════════════════════════════
# 模型加载
# ═══════════════════════════════════════════════════════════════

MODEL_PATH = "tag_embedding.pt"

_device = torch.device("cpu")
_model_loaded = False
_embed_dim = 32

try:
    # torch.load: 从磁盘加载之前 train.py 保存的 state_dict
    # map_location='cpu': 如果模型是在 GPU 上训练的，自动转为 CPU 张量
    checkpoint = torch.load(MODEL_PATH, map_location=_device)

    # 重建模型结构
    _model = TagEmbedding(num_tags=NUM_TAGS,
                          embed_dim=checkpoint.get("embed_dim", 32))

    # 把保存的权重"灌"进模型
    #   复制到模型的对应参数中
    _model.load_state_dict(checkpoint["model_state_dict"])

    # model.eval(): 切换到评估模式
    _model.eval()

    _embed_dim = checkpoint.get("embed_dim", 32)
    _model_loaded = True

except FileNotFoundError:
    # 模型文件不存在 → 静默处理，后续回退到 Dice 系数
    pass


# ═══════════════════════════════════════════════════════════════
# 推理函数
# ═══════════════════════════════════════════════════════════════

def embed_tags(tags: set[str]) -> torch.Tensor:
    """
    标签集合 → 32维稠密嵌入。

    这是推理的核心: 一道题经过 LLM 打标得到标签集合，
    然后调用这个函数得到嵌入向量。
    之后用余弦相似度比较两个嵌入向量。

    返回: (32,) 一维张量
    """
    # 1. 标签集合 → 213维多热向量: {"序列","数组"} → [1,1,0,...,0]
    vec = tags_to_multihot(tags)
    # 2. 加 batch 维度: (213,) → (1, 213)
    #    unsqueeze(0): 在第0维前插入一个新维度
    #    model.forward 期望输入是 (batch, 213)，哪怕只有1个样本
    vec = vec.unsqueeze(0)
    # 3. 前向传播
    with torch.no_grad(): emb = _model(vec)
    # 4. 去掉 batch 维度
    return emb.squeeze(0)


def get_ranked_list(query_tags: set[str]) -> list[tuple[str, float, set[str]]]:
    """
    检索: 给定查询的标签集，返回库中所有题按相似度排序的列表。
    """
    items = list(tool.list_all().items())
    if not items:
        return []

    # ── 嵌入模式 ──
    if _model_loaded:
        # 查询嵌入
        query_emb = embed_tags(query_tags)   # (32,)

        # 库中所有题的嵌入
        all_vecs = torch.stack(
            [tags_to_multihot(info["tags"]) for _, info in items]
        )
        with torch.no_grad():
            all_embs = _model(all_vecs)   # (N, 213) → (N, 32)

        # 余弦相似度: cos(A,B) = (A·B) / (||A|| * ||B||)
        #   torch.nn.functional.cosine_similarity 直接算
        #   值域 [-1, 1]: 1=方向完全相同, 0=正交(无关), -1=方向相反
        cos_sim = torch.nn.functional.cosine_similarity(
            all_embs, query_emb.unsqueeze(0), dim=1
        )  # (N,)

        array = []
        for i, (idx, info) in enumerate(items):
            sim = cos_sim[i].item()
            array.append((idx, sim, info["tags"]))
        array.sort(key=lambda x: -x[1])     # 相似度降序
        return array


# ═══════════════════════════════════════════════════════════════
# 命令行入口
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # 1. 从 stdin 读入题目文本
    statement = sys.stdin.read()
    if not statement.strip():
        print("用法: python main.py < problem.txt")
        print("      echo 'N cows...' | python main.py")
        sys.exit(1)

    print("Input Received!")

    # 2. LLM 打标 —— 调用 AI 
    tags,_ = tool.convert(statement)
    print(f"tags ({len(tags)}): {' '.join(sorted(tags))}")

    # 3. 检查数据库状态
    n_stored = tool.count_all()
    print(f"数据库中有 {n_stored} 道题")
    if _model_loaded: print(f"嵌入模型: {MODEL_PATH} (213 → {_embed_dim})")

    if n_stored == 0:
        print("\n 数据库为空！")
        sys.exit(0)

    # 4. 检索排名
    array = get_ranked_list(tags)

    Mean=sum(x[1] for x in array)/len(array)
    StDev=sqrt(sum((x[1]-Mean)**2 for x in array)/(len(array)-1))

    # 5. 打印结果
    print(f"Mean={Mean}")
    print(f"Standard Deviation={StDev}")
    print(f"\n{'='*35} Ranking {'='*35}")

    max_name_len = max((len(item[0]) for item in array), default=20)
    name_width = max(20, max_name_len + 2)

    for i in range(min(len(array), 200)):
        name = array[i][0]
        sim = array[i][1]
        z_score=(sim-Mean)/StDev

        marker = "★" if z_score>2 else " "

        tags_str = ' '.join(sorted(array[i][2]))
        if len(tags_str) > 60:
            tags_str = tags_str[:60] + "..."

        print(f"{i+1:<4} {marker} {name:<{name_width}} {sim:>8.3f} {z_score:>8.3f}  {tags_str}")

    print(list(x[1] for x in array))
