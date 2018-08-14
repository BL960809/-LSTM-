import tensorflow as tf


class LSTMNet(object):
    def __init__(self, n_label, embeddings, embeddingSize, rnnSize, max_sequence_length, learning_rate):
        self.n_label = n_label
        self.embeddings = embeddings
        self.embeddingSize = embeddingSize
        self.rnnSize = rnnSize
        self.max_sequence_length = max_sequence_length
        self.learning_rate = learning_rate

        self.x_data = tf.placeholder(tf.int32, [None, self.max_sequence_length])
        self.y_output = tf.placeholder(tf.int32, [None, self.n_label])

        # 设置 embedding 层
        idx = tf.Variable(tf.to_float(self.embeddings))
        embedding_output = tf.nn.embedding_lookup(idx, self.x_data)

        # 设置 LSTM 结构
        cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.rnnSize)
        output, state = tf.nn.dynamic_rnn(cell, embedding_output, dtype=tf.float32)

        # 前向传播结构
        output = tf.transpose(output, [1, 0, 2])
        last = tf.gather(output, int(output.get_shape()[0]) - 1)

        weight = tf.Variable(tf.truncated_normal([self.rnnSize, self.n_label], stddev=0.1))
        bias = tf.Variable(tf.constant(0.1, shape=[self.n_label]))
        self.logits_out = tf.nn.softmax(tf.matmul(last, weight) + bias)

        # 损失函数
        losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits_out, labels=self.y_output)
        self.loss = tf.reduce_mean(losses)

        # 优化器
        optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
        self.train_step = optimizer.minimize(self.loss)


