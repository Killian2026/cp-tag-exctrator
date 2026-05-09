"""
  Linear + ReLU，单层神经网络
"""

import torch
import torch.nn as nn


class TagEmbedding(nn.Module):
    def __init__(self, num_tags: int = 213, embed_dim: int = 32):
        """
        构造函数

        Args:
            num_tags: 输入维度 = 标签总数
            embed_dim: 输出维度 = 嵌入空间维度
        """
        super().__init__()
        self.num_tags = num_tags
        self.embed_dim = embed_dim

        # ── 第一层: 线性变换 ──
        self.linear = nn.Linear(num_tags, embed_dim, bias=True)

        # ── 激活函数: ReLU ──
        self.activation = nn.ReLU()

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """
        权重初始化

        Xavier初始化 (Glorot初始化): 让输入和输出的方差保持一致。  

        偏置初始化为0
        """
        nn.init.xavier_uniform_(self.linear.weight)  # W ~ U(-a, +a)
        nn.init.zeros_(self.linear.bias)              # b = 0

    def forward(self, multi_hot: torch.Tensor) -> torch.Tensor:
        """
        前向传播: Linear → L2归一化

        L2归一化把嵌入投影到单位球面上:
          - 训练: L2距离 ≈ 余弦距离，与推理一致
          - 推理: 余弦相似度直接可用
          - 不需要ReLU (归一化本身约束了向量大小)
        """
        x = self.linear(multi_hot)
        x = torch.nn.functional.normalize(x, p=2, dim=1)
        return x

    def normalize_embeddings(self):
        """
        L2归一化权重，变成长度为1的向量
        """
        with torch.no_grad():
            norm = self.linear.weight.data.norm(dim=1, keepdim=True)

            self.linear.weight.data = self.linear.weight.data / (norm + 1e-8)

    def get_tag_embeddings(self) -> torch.Tensor:
        """
        获取每个标签的嵌入向量（用于分析和可视化）。

        返回维度：(标签数, 向量维度)
        """
        return self.linear.weight.data.T
