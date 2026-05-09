"""
数据库模块 - JSON 文件存储, 可直接打开 problems.json 查看数据
格式: {"题号": {"statement": "题面", "tags": ["标签1", "标签2"]}}
"""

import json
import os

DB_PATH = "problems.json"


class _DB:
    """题号 → {信息} 的字典数据库, 自动持久化到 JSON 文件"""

    def __init__(self):
        self._load()

    def _load(self):
        if os.path.exists(DB_PATH):
            with open(DB_PATH, "r", encoding="utf-8") as f:
                self._data = json.load(f)
        else:
            self._data = {}

    def _save(self):
        with open(DB_PATH, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)

    # ---- 像 dict 一样使用 ----

    def __setitem__(self, pid: str, info: dict):
        """db['题号'] = {'statement': ..., 'tags': {...}, 'raw_output': '...'}"""
        tags = info.get("tags", set())
        self._data[pid] = {
            "statement": info["statement"],
            "tags": sorted(tags),
            "raw_output": info.get("raw_output", ""),
        }
        self._save()

    def __getitem__(self, pid: str) -> dict:
        """p = db['题号']; p['tags'] 是集合, p['raw_output'] 是原始AI输出"""
        v = self._data[pid]
        return {
            "statement": v["statement"],
            "tags": set(v["tags"]),
            "raw_output": v.get("raw_output", ""),
        }

    def __contains__(self, pid: str) -> bool:
        return pid in self._data

    def __len__(self) -> int:
        return len(self._data)

    def __delitem__(self, pid: str):
        """del db['题号']"""
        del self._data[pid]
        self._save()

    def pop(self, pid: str):
        """删除并返回题目, 不存在返回 None"""
        if pid not in self._data:
            return None
        v = self._data.pop(pid)
        self._save()
        return {
            "statement": v["statement"],
            "tags": set(v["tags"]),
            "raw_output": v.get("raw_output", ""),
        }

    def clear(self):
        """删除全部题目"""
        self._data.clear()
        self._save()

    def get(self, pid: str):
        """安全获取, 不存在返回 None"""
        try:
            return self[pid]
        except KeyError:
            return None

    def items(self):
        """遍历所有: for pid, info in db.items(): ..."""
        for pid, v in self._data.items():
            yield pid, {
                "statement": v["statement"],
                "tags": set(v["tags"]),
                "raw_output": v.get("raw_output", ""),
            }

    def search(self, tag: str) -> dict:
        """按标签搜索, 返回 {题号: {信息}}"""
        result = {}
        for pid, v in self._data.items():
            if tag in v["tags"]:
                result[pid] = {
                    "statement": v["statement"],
                    "tags": set(v["tags"]),
                    "raw_output": v.get("raw_output", ""),
                }
        return result


# 全局单例, 别处直接 import 就能用
db = _DB()
