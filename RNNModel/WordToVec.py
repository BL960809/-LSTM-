import re
import numpy as np
from gensim.models import word2vec


# 定义文本清洗的功能函数
def clean_text(text_string):
    # 剔除文本中的网址、ASCII 码和数字
    text_string = re.sub(r'www\.[a-zA-Z0-9\.\?/&\=\:]+|(http\://[a-zA-Z0-9\.\?/&\=\:]+)|&.*?;|[0-9]+', ' ', text_string)
    # 剔除文本中的特殊字符
    text_string = re.sub(r'[\s+\!\/_,$%^*(+\")]+|[+——(),:;?【】“”！，。？、~@#￥%……&*（）]+', ' ', text_string)
    text_string = " ".join(text_string.split())
    text_string = text_string.lower()

    return text_string


# 使用第三方停词包去掉停词
def segment(content, stopwords):
    if isinstance(content, str) is False:
        content = ' '.join(content)

    text = content.split(' ')  # 分词,默认是精确分词
    result = []
    for word in text:
        word = word.strip('.')
        word = word.strip("'")
        if len(word) != 0 and word != '-' and not stopwords.__contains__(word):  # 去掉在停用词表中出现的内容
            result.append(word)

    return result


# 生成词向量字典
def build_word_dict(textPath):
    ## 计算词向量
    sentences = word2vec.Text8Corpus(textPath)
    model = word2vec.Word2Vec(sentences, size=50, window=10, sg=1, hs=1, iter=10, min_count=1)

    ## 将词向量转化为字典输出
    vocab = model.wv.vocab
    index = []
    for word in vocab:
        index.append(model[word])

    a = np.array(index, dtype=np.float)

    word_vector = {}
    i = 0
    for word in vocab:
        word_vector[word] = list(a[i])
        i += 1

    return word_vector


