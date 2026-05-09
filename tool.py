import LLM
import db

DB = db.db  # dict 式的数据库

def convert(statement: str) -> tuple[set,str]:
    answer = LLM.call_AI(statement)
    valid_tags = set(LLM.DICT.keys())  # 词表中的 213 个合法标签
    result = set()
    for line in answer.splitlines():
        line = line.strip()
        if not line or '=' not in line: continue
        key, val = line.split('=', 1)
        key = key.strip()
        if val.strip() != '1': continue
        result.add(key)
    return result,answer

def find_problem(problem_id: str):
    """按题号查找"""
    return DB.get(problem_id)

def add_problem(problem_id: str, statement: str):
    """添加题目, AI 自动打标"""
    if find_problem(problem_id)!=None: return

    tags, raw_output=convert(statement)
    DB[problem_id] = {"statement": statement, "raw_output":raw_output, "tags": tags}



def list_all():
    """列出所有题目"""
    return dict(DB.items())

def show_all():
    for idx,info in list_all().items():
        print(f"{idx}   {' '.join(info['tags'])}")


def search_by_tag(tag: str):
    """按标签搜索"""
    return DB.search(tag)


def delete_problem(problem_id: str):
    """按题号删除"""
    return DB.pop(problem_id)


def clear_all():
    """清空全部题目"""
    DB.clear()


def count_all():
    """总题数"""
    return len(DB)

if __name__=="__main__":
    show_all()
