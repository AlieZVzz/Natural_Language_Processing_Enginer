# import numpy as np
# from scipy.spatial.distance import pdist#直接调包可以计算JC值 :需要两个句子长度一样；所以暂时不用
import jieba


# from nltk import word_tokenize 英文分词

def Jaccrad(model, reference):  # terms_reference为源句子，terms_model为候选句子
    terms_reference = jieba.cut(reference)  # 默认精准模式
    terms_model = jieba.cut(model)
    grams_reference = set(terms_reference)  # 去重；如果不需要就改为list
    grams_model = set(terms_model)
    temp = 0
    for i in grams_reference:
        if i in grams_model:
            temp = temp + 1
    fenmu = len(grams_model) + len(grams_reference) - temp  # 并集
    jaccard_coefficient = float(temp / fenmu)  # 交集
    return jaccard_coefficient


a = "香农在信息论中提出的信息熵定义为自信息的期望"
b = "信息熵作为自信息的期望"
jaccard_coefficient = Jaccrad(a, b)
print(jaccard_coefficient)
