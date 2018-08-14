import re
import time
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


def load_data(data_path):
    IDs, texts = [], []
    with open(data_path, 'r', encoding='utf8') as file:
        for line in file.readlines():
            article = line.split('\t')
            IDs.append(article[0])
            texts.append(article[7:])

    return IDs, texts


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

    result = ' '.join(str(e) for e in result)

    return result


# 文件路径
data_path = '乐惠国际/Data/'
start_time = time.time()      # 计时开始

# 外部导入第三方停词包
stopwords = [line.strip() for line in open('乐惠国际/Data/stop_words.txt', encoding='utf-8').readlines()]

# 加载文件
IDs, texts = load_data(data_path)
for i in range(len(texts)):
    texts[i] = ' '.join(str(e.strip()) for e in texts[i])

# 清洗文本
texts = [clean_text(x) for x in texts]
texts = [segment(x, stopwords) for x in texts]

# 初始化
transformer = TfidfTransformer()
vectorizer = CountVectorizer()

# 计算权重
tfidf = transformer.fit_transform(vectorizer.fit_transform(texts))
words = vectorizer.get_feature_names()
weights = tfidf.toarray()

keyword_dict = {}
for i in range(len(IDs)):
    words_weight = zip(words, list(weights[i]))
    words_weight = sorted(words_weight, key=lambda x: x[1], reverse=True)
    keywords = list(zip(*words_weight))

    keyword_dict[IDs[i]] = keywords[0][:10]

end_time = time.time()      # 计时结束
print('权重计算完毕，总用时 %.3f 秒' % (end_time-start_time))


# 输出权重文件
dataframe = pd.DataFrame(keyword_dict)
dataframe.T.to_csv('乐惠国际/Output/Keywords.csv', header=False)


