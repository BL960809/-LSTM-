import os
import time
import pandas as pd
import tensorflow as tf
import numpy as np
import sys
sys.path.append('D:/lehui/乐惠国际/文本标签预测模型')

import LoadData
import WordToVec

# 参数设置
n_label = 3
max_sequence_length = 20            # 语句最大长度为50个单词，超过部分会被截取掉，不够的部分用0填充
rnn_size = 40                       # rnn模型由20个单元组成,一个cell中神经元的个数
embedding_size = 50                 # 每个单词会被嵌套在长度为50的词向量中
learning_rate = 0.001

# 设置文件路径
data_dir = '乐惠国际/Data/'       # 文件所在目录
cut_name = 'cut.txt'                   # 分词文件名
embedding_name = 'embedding.txt'       # 词向量文件名
word2vec_model = 'word2vec_model'      # 词向量模型
test_data_file = 'lehui0801.txt'
model_file = '乐惠国际/newmodel/savedModel'
start_time = time.time()

# 外部导入第三方停词包
stopwords = [line.strip() for line in open('乐惠国际/Data/stop_words.txt', encoding='utf-8').readlines()]

# load test.txt
article_IDs, sentence_IDs, texts_original = [], [], []
with open(os.path.join(data_dir, test_data_file), 'r', encoding='gbk') as file_conn:
    for row in file_conn:
        tmp = row.split('\t')
        sentence_IDs.append(tmp[1])
        article_IDs.append(tmp[2])
        texts_original.append(tmp[3])

# 文本清洗
article_IDs = [x.strip() for x in article_IDs]
sentence_IDs = [x.strip() for x in sentence_IDs]
texts = [WordToVec.clean_text(x) for x in texts_original]

print('加载词向量文件...')
embedding, word2idx = LoadData.loadEmbedding(data_dir + 'word2vec.csv')
print('加载完成')

print('对文本进行清洗...')
sample_texts = [LoadData.sentenceToIndex(x, word2idx, max_sequence_length, stopwords) for x in texts]
sample_texts = np.array(sample_texts)

print('加载模型结构...')
# 定义 RNN 模型
x_data = tf.placeholder(tf.int32, [None, max_sequence_length])
y_output = tf.placeholder(tf.int32, [None, n_label])

# 设置 embedding 层
idx = tf.Variable(tf.to_float(embedding))
embedding_output = tf.nn.embedding_lookup(idx, x_data)

# 设置 LSTM 结构
cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=rnn_size)
output, state = tf.nn.dynamic_rnn(cell, embedding_output, dtype=tf.float32)

# 前向传播结构
output = tf.transpose(output, [1, 0, 2])
last = tf.gather(output, int(output.get_shape()[0]) - 1)

weight = tf.Variable(tf.truncated_normal([rnn_size, n_label], stddev=0.1))
bias = tf.Variable(tf.constant(0.1, shape=[n_label]))
logits_out = tf.nn.softmax(tf.matmul(last, weight) + bias)

print('开始预测...')
with tf.Session() as sess:
    saver = tf.train.Saver()

    # load the model
    saver.restore(sess, model_file)

    predict = sess.run(logits_out, feed_dict={x_data: sample_texts})
    result = sess.run(tf.arg_max(predict, 1))

end_time = time.time()
print('预测完成，总用时 %.2f 秒' %(end_time - start_time))

print('输出预测结果...')
df = pd.DataFrame({'article_ID': article_IDs, 'sentence_ID:': sentence_IDs, 'label': result, 'text': texts_original})
cols = ['article_ID', 'sentence_ID:', 'text', 'label']
df = df.ix[:, cols]
df.to_csv('乐惠国际/Output/Prediction4lehui_3classes.csv', sep='\t', index=None, header=0)
print('Finish!')

