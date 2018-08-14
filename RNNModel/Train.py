import os
import time
import csv
import numpy as np
import tensorflow as tf
import sys
import matplotlib.pyplot as plt
sys.path.append('D:/lehui/乐惠国际/文本标签预测模型')

import LoadData
import WordToVec
from LSTMModel import LSTMNet


# 参数设置
epochs = 500
n_label = 3
batch_size = 40
max_sequence_length = 20            # 语句最大长度为50个单词，超过部分会被截取掉，不够的部分用0填充
rnn_size = 40                       # rnn模型由20个单元组成,一个cell中神经元的个数
embedding_size = 50                 # 每个单词会被嵌套在长度为50的词向量中
learning_rate = 0.001

# 设置文件路径
data_dir = '乐惠国际/Data/'       # 文件所在目录
cut_name = 'cut.txt'                   # 分词文件名
embedding_name = 'embedding.txt'       # 词向量文件名
train_data_file = 'Train4lehui_3class.txt'
test_data_file = 'Test4lehui_3class.txt'
model_file = '乐惠国际/newmodel/savedModel'

# 外部导入第三方停词包
stopwords = [line.strip() for line in open('乐惠国际/Data/stop_words.txt', encoding='utf-8').readlines()]

# load train.txt
train_sentenceIDs, train_articleIDs, train_text_data, train_label = [], [], [], []
with open(os.path.join(data_dir, train_data_file), 'r', encoding='utf-8') as file_conn:
    for row in file_conn:
        tmp = row.split('\t')
        train_sentenceIDs.append(tmp[1])
        train_articleIDs.append(tmp[2])
        train_text_data.append(tmp[3])
        train_label.append(tmp[-1].strip())

# load test.txt
test_sentenceIDs, test_articleIDs, test_text_data, test_label = [], [], [], []
with open(os.path.join(data_dir, test_data_file), 'r', encoding='utf-8') as file_conn:
    for row in file_conn:
        tmp = row.split('\t')
        test_sentenceIDs.append(tmp[1])
        test_articleIDs.append(tmp[2])
        test_text_data.append(tmp[3])
        test_label.append(tmp[-1].strip())

label = test_label + train_label
text = test_text_data + train_text_data

# 文本清洗
text = [WordToVec.clean_text(x) for x in text]

try:
    print('加载词向量文件...')
    embedding, word2idx = LoadData.loadEmbedding(data_dir + 'word2vec.csv')

    print('加载完成')
except IOError:
    choice = input('未找到词向量文件，是否重新训练词向量？(y/n)')
    if choice.strip().lower() == 'y':
        print('正在进行分词...')
        words = LoadData.segment(text, stopwords)
        with open(data_dir + cut_name, 'w', newline='', encoding='utf-8') as cutfile:
            for word in words:
                cutfile.write(word + ' ')

        print('正在训练词向量...')
        word_vect = WordToVec.build_word_dict(data_dir + cut_name)
        with open(data_dir + 'word2vec.csv', mode='w', newline='', encoding='utf-8') as csv_file:
            writer = csv.writer(csv_file, delimiter=' ')
            for key in list(word_vect.keys()):
                tmp = [key] + list(word_vect[key])
                writer.writerow(tmp)

        print('加载词向量文件...')
        embedding, word2idx = LoadData.loadEmbedding(data_dir + 'word2vec.csv')

        print('加载完成')
    elif choice.strip().lower() == 'n':
        print('请将词向量文件添加至路径...')
    else:
        print('无效的输入 \n')

# 文本进行 embedding 编码
text = [LoadData.sentenceToIndex(x, word2idx, max_sequence_length, stopwords) for x in text]

# 划分训练集、测试集
text_array = np.array(text)
label_array = np.array(label)

x_test, x_train = text_array[:47], text_array[47:]
y_test, y_train = label_array[:47], label_array[47:]

train_loss = []
test_loss = []
train_accuracy = []
test_accuracy = []

print('正在加载神经网络模型...')
# 加载 lstm 模型
lstm = LSTMNet(n_label, embedding, embedding_size, rnn_size, max_sequence_length, learning_rate)

print('开始训练...')
# 启用 Session
with tf.Session() as sess:
    # 初始化 Session
    sess.run(tf.global_variables_initializer())
    # 初始化 Saver
    saver = tf.train.Saver()
    # 开始计时
    start_time = time.time()

    y_train_tmp = sess.run(tf.one_hot(y_train, depth=n_label))
    y_test_tmp = sess.run(tf.one_hot(y_test, depth=n_label))

    # 开始训练
    for epoch in range(epochs):
        # 使用 batch 对数据进行分批训练
        num_batches = int(len(x_train)/batch_size)+1
        for i in range(num_batches):
            min_ix = i*batch_size
            max_ix = np.min([len(x_train), ((i+1)*batch_size)])
            x_train_batch = x_train[min_ix:max_ix]
            y_train_batch = y_train_tmp[min_ix:max_ix]

            #  前向传播
            train_dict = {lstm.x_data: x_train_batch, lstm.y_output: y_train_batch}
            sess.run(lstm.train_step, feed_dict=train_dict)

        #  计算 Loss 和 Acc
        temp_train_loss = sess.run(lstm.loss, feed_dict={lstm.x_data: x_train, lstm.y_output: y_train_tmp})
        train_loss.append(temp_train_loss)

        #  输出每一次迭代的状态
        test_dict = {lstm.x_data: x_test, lstm.y_output: y_test_tmp}
        temp_test_loss, prediction = sess.run([lstm.loss, lstm.logits_out], feed_dict=test_dict)
        test_loss.append(temp_test_loss)

        if (epoch + 1) % 100 == 0:
            print('Epoch: {}, Train Loss: {:.4}, Test Loss: {:.4}'.format(epoch + 1, temp_train_loss, temp_test_loss))

    # 结束计时
    end_time = time.time()

    # 持久化模型
    saver.save(sess, model_file)

    # 输出预测结果
    print('------- 检验模型效果 -------')
    pred = sess.run(tf.argmax(prediction, 1))

    def addone(x):
        # 对应回句子所在的行数

        return x + 1

    print('模型预测出为 0 的句子：', list(map(addone, np.where(pred == 0)[0].tolist())))
    print('验证集中实际为 0 的句子:', list(map(addone, np.where(y_test == '0')[0].tolist())))

    print('模型预测出为 1 的句子：', list(map(addone, np.where(pred == 1)[0].tolist())))
    print('验证集中实际为 1 的句子:', list(map(addone, np.where(y_test == '1')[0].tolist())))

    print('模型预测出为 2 的句子：', list(map(addone, np.where(pred == 2)[0].tolist())))
    print('验证集中实际为 2 的句子:', list(map(addone, np.where(y_test == '2')[0].tolist())))


# 绘制折线图
ite = [x+1 for x in range(epochs)]
y_train_plt = train_loss
y_test_plt = test_loss

plt.plot(ite, y_train_plt, label=u'Loss for train')
plt.plot(ite, y_test_plt, label=u'Loss for test')
plt.legend()  # 让图例生效


