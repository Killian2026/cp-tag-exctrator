"""
数据集工具

多热编码
"""

import torch
from torch.utils.data import Dataset
from TagDict import DICT


# ═══════════════════════════════════════════════════════════════
# 全局常量 — 程序启动时算一次，之后全部复用
# ═══════════════════════════════════════════════════════════════

NUM_TAGS = len(DICT)  # 213 — 标签总数

# 标签名 → 向量中的位置编号
# 例: TAG_TO_IDX["序列"] = 0, TAG_TO_IDX["数组"] = 1, ...
# 这是从 TagDict.py 里每个标签的 'id' 字段读的
TAG_TO_IDX: dict[str, int] = {tag: info["id"] for tag, info in DICT.items()}

# 向量中的位置编号 → 标签名（反向查询，调试用）
IDX_TO_TAG: dict[int, str] = {info["id"]: tag for tag, info in DICT.items()}

# 确认 id 是连续的 0~212，没有跳号
# 如果不连续，多热向量的某个位置永远为0，浪费维度
assert len(TAG_TO_IDX) == NUM_TAGS
assert set(TAG_TO_IDX.values()) == set(range(NUM_TAGS)), \
    f"TagDict id 不连续! 期望 0-{NUM_TAGS-1}，实际 {sorted(TAG_TO_IDX.values())}"


# ═══════════════════════════════════════════════════════════════
# 多热向量转换
# ═══════════════════════════════════════════════════════════════

def tags_to_multihot(tags: set[str]) -> torch.Tensor:
    """
    {"前缀和", "数组", "差分"}  →  [1, 1, 0, ..., 1, ..., 0]

    算法:
      1. 创建一个全0的213维向量
      2. 遍历输入集合中的每个标签
      3. 查 TAG_TO_IDX 找到这个标签对应的位置
      4. 把那个位置设为 1.0

    返回值是浮点向量(float32)，不是整数向量。
    原因: nn.Linear 要求输入是 float，整数会报错。
    """
    # torch.zeros(N): 创建一个长度为 N、全为 0.0 的向量
    vec = torch.zeros(NUM_TAGS)

    for tag in tags:
        if tag in TAG_TO_IDX:           # 防止未知标签（LLM 偶尔输出不在词表里的词）
            idx = TAG_TO_IDX[tag]        # 标签对应的位置编号
            vec[idx] = 1.0               # 该位置设为 1

    return vec


def multihot_to_tags(vec: torch.Tensor, threshold: float = 0.5) -> set[str]:
    """
    [0, 1, 0, ..., 1, ...]  →  {"数组", "前缀和", ...}

    正向的逆操作，用于:
      - 调试: 看看模型输出的多热向量对应哪些标签
      - 推理: 如果模型输出连续值（非严格0/1），用阈值决定哪些标签算"有"

    threshold=0.5 的含义: 值 ≥ 0.5 就算这个标签存在。
    对于严格的多热向量（只有0和1），阈值0.5没问题。
    """
    tags = set()
    # (vec > threshold) 返回布尔张量，.nonzero() 返回 True 的位置索引
    for idx in (vec > threshold).nonzero(as_tuple=True)[0]:
        idx = idx.item()   # 从 0维tensor 转成 Python int
        if idx in IDX_TO_TAG:
            tags.add(IDX_TO_TAG[idx])
    return tags


# ═══════════════════════════════════════════════════════════════
# PyTorch Dataset — 训练数据的标准化接口
# ═══════════════════════════════════════════════════════════════

class ContrastiveDataset(Dataset):
    """
    继承 torch.utils.data.Dataset —— PyTorch 的"数据集标准接口"。
    """

    def __init__(self, data: dict):
        """
        把 JSON 数据转换成神经网络能吃的张量列表。

        输入 data 的结构:
          {
            "positive_pairs": [
              {"tags_a": ["序列", "数组"], "tags_b": ["序列", "数组", "模拟"], "label": 1},
              ...
            ],
            "negative_pairs": [
              {"tags_a": [...], "tags_b": [...], "label": 0},
              ...
            ],
            "hard_negative_pairs": [...]
          }
        """
        self.pairs = []

        # 先收集 originals 的标签，供后续通过 problem_id 查找
        originals_tags = {}
        if "originals" in data:
            for oid, info in data["originals"].items():
                if "expected_tags" in info:
                    originals_tags[oid] = set(info["expected_tags"])

        # 遍历三类配对（正样本、负样本、困难负样本）
        # 它们的区别只在于 label 和标签重叠程度，处理逻辑完全一样
        for pair_list_name in ["positive_pairs", "negative_pairs", "hard_negative_pairs"]:
            for p in data.get(pair_list_name, []):
                # 解析两边的标签集合
                tags_a = self._resolve_tags(p, "a", originals_tags)
                tags_b = self._resolve_tags(p, "b", originals_tags)

                # 如果任一侧解析失败（标签为空且找不到对应problem），跳过这个pair
                if tags_a is None or tags_b is None:
                    continue

                # 标签 → 多热向量
                vec_a = tags_to_multihot(tags_a)  # (213,) 浮点向量
                vec_b = tags_to_multihot(tags_b)  # (213,) 浮点向量

                # label 存为 float32 张量
                # 为什么用 float 而不是 int？
                #   Contrastive Loss 里 label 直接乘 dist²，
                #   float×float 不需要类型转换，int×float 会触发警告
                label = torch.tensor(p["label"], dtype=torch.float32)

                self.pairs.append({
                    "pair_id": p.get("pair_id", "?"),
                    "vec_a": vec_a,
                    "vec_b": vec_b,
                    "label": label,
                })

    def _resolve_tags(self, p: dict, side: str, originals_tags: dict) -> set[str] | None:
        """
        从 pair 中解析某一侧(side="a"或"b")的标签集合。

        支持两种数据格式:
          1. 直接给标签: {"tags_a": ["序列", "数组"], ...}
          2. 通过problem_id间接查: {"id_a": "24-1-Bronze-1", ...}
             此时从 originals 映射表中查找这道题的 expected_tags

        返回 None 表示无法解析（外部会跳过这个pair）。
        """
        # 优先: tags_a / tags_b 字段直接指定
        key = f"tags_{side}"   # "tags_a" 或 "tags_b"
        if key in p and p[key]:
            return set(p[key])

        # 备选: id_a / id_b 或 original_id 字段
        if side == "a":
            oid = p.get("id_a") or p.get("original_id")
        else:
            oid = p.get("id_b") or p.get("original_id")

        if oid and oid in originals_tags:
            return originals_tags[oid]

        return None

    def __len__(self):
        """返回数据集大小。DataLoader 用它来规划每个epoch。"""
        return len(self.pairs)

    def __getitem__(self, idx: int):
        """
        必须实现: 返回第 idx 个样本。

        DataLoader 调用流程:
          for indices in batch_sampler:
            for i in indices:
              data = dataset[i]  ← 这里触发 __getitem__(i)
            batch = collate_fn(all_data)  ← 自动堆叠成 (batch, 213)

        返回的4个张量会被 DataLoader 自动堆叠:
          [vec_a_0, vec_a_1, ..., vec_a_{B-1}] → (B, 213)
          [vec_b_0, vec_b_1, ..., vec_b_{B-1}] → (B, 213)
          [label_0, ..., label_{B-1}]           → (B,)
          [pid_0, ..., pid_{B-1}]               → tuple of strings
        """
        p = self.pairs[idx]
        return p["vec_a"], p["vec_b"], p["label"], p["pair_id"]
