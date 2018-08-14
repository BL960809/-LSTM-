import re
import time
from collections import defaultdict


def segment(content, stopwords):
    if isinstance(content, str) is False:
        content = ' '.join(content)

    text = content.split(' ')  # 分词,空格作为分隔符
    result = []
    for word in text:
        word = word.strip('.')
        word = word.strip("'")
        if len(word) != 0 and word != '-' and not stopwords.__contains__(word):  # 去掉在停用词表中出现的内容
            result.append(word)

    return result


# 加载数据集文件
def load_data(filepath):
    date = time.strftime('_%Y%m%d')
    text_data = []
    with open(filepath + 'feedback' + date + '.txt', 'r', encoding='utf8') as file_conn:
        for row in file_conn:
            text_data.append(row)

    text_data = [x.split('\t') for x in text_data if len(x)>=1]
    [label, centent] = [list(x) for x in zip(*text_data)]

    return label, centent


# 加载词向量文件
def loadEmbedding(filename):
    embeddings = []
    word2idx = defaultdict(list)
    with open(filename, mode="r", encoding="utf-8") as rf:
        for line in rf:
            arr = line.split(" ")
            embedding = [float(val) for val in arr[1:]]
            word2idx[arr[0]] = len(word2idx)
            embeddings.append(embedding)

    return embeddings, word2idx


# 对文本进行 one-hot 编码
def sentenceToIndex(sentence, word2idx, maxLen, stopwords):
    unknown = word2idx.get("UNKNOWN", 0)
    num = word2idx.get("NUM", len(word2idx) - 1)
    index = [unknown] * maxLen
    i = 0
    for word in segment(sentence, stopwords):
        if word in word2idx:
            index[i] = word2idx[word]
        else:
            if re.match(r"\d+", word):
                index[i] = num
            else:
                index[i] = unknown
        if i >= maxLen - 1:
            break
        i += 1

    return index
