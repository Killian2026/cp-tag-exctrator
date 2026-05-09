from TagDict import DICT
from math import sqrt

def weight(x):
    if x not in DICT:
        return 0
    return DICT[x].get("Weight", 1.0)

def Jaccard(A: set, B: set) -> float:
    """
    计算两个集合 Jaccard 相似度
    """
    if A==B: return 1
    sumN=sum(weight(x) for x in A&B)
    sumU=sum(weight(x) for x in A|B)
    return sumN/sumU

def Cosine(A: set, B:set) -> float:
    """
    计算两个集合余弦相似度
    """
    if A==B: return 1
    sumSq= sum(weight(x)**2 for x in A&B)
    sumSqA=sum(weight(x)**2 for x in A)
    sumSqB=sum(weight(x)**2 for x in B)

    return sumSq/sqrt(sumSqA*sumSqB)

def Dice(A: set, B:set) -> float:
    """
    计算两个集合 Dice 系数
    """
    if A==B: return 1
    sumU=sum(weight(x) for x in A&B)
    sumA=sum(weight(x) for x in A)
    sumB=sum(weight(x) for x in B)

    return 2*sumU/(sumA+sumB)
