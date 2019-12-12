import tensorflow as tf

SRC_TRAIN_DATA = r"D:\1研究生\机器学习\tensorflow\《TensorFlow实战Google深度学习框架(第2版)》中文版PDF和源代码\源代码\Chapter09\train.en"        # 源语言输入文件。
TRG_TRAIN_DATA =  r"D:\1研究生\机器学习\tensorflow\《TensorFlow实战Google深度学习框架(第2版)》中文版PDF和源代码\源代码\Chapter09\train.zh"        # 目标语言输入文件。
CHECKPOINT_PATH =  r"D:\1研究生\机器学习\tensorflow\《TensorFlow实战Google深度学习框架(第2版)》中文版PDF和源代码\源代码\Chapter09\seq2seq_ckpt"   # checkpoint保存路径。

HIDDEN_SIZE = 1024
BATCH_SIZE = 100
SRC_VOCAB_SIZE = 10000
TRG_VOCAB_SIZE = 4000
NUM_LAYERS = 2
NUM_EPOCH = 5
KEEP_PROB = 0.8
MAX_GRAD_NORM = 5
SHARE_EMB_AND_SOFTMAX = True
MAX_LEN = 50
SOS_ID = 1

def MakeDataset(file_path):
    dataset = tf.data.TextLineDataset(file_path)
    dataset = dataset.map(lambda string: tf.string_split([string]).values)
    dataset = dataset.map(lambda string: tf.string_to_number(string,tf.int32))
    dataset = dataset.map(lambda x: (x,tf.size(x)))
    return dataset

def MakeSrcTrgDataset(src_path,trg_path,batch_size):
    src_data = MakeDataset(src_path)
    trg_data = MakeDataset(trg_path)
    dataset = tf.data.Dataset.zip((src_data,trg_data))
    def FilterLength(src_tuple, trg_tuple):
        ((src_input,src_len),(trg_input,trg_len)) = (src_tuple,trg_tuple)
        src_len_ok = tf.logical_and(tf.greater(src_len,1),tf.less_equal(src_len,MAX_LEN))
        trg_len_ok = tf.logical_and(tf.greater(trg_len,1),tf.less_equal(trg_len,MAX_LEN))
        return tf.logical_and(src_len_ok,trg_len_ok)
    dataset.filter(FilterLength)

    def MakeTrgInput(src_tuple,trg_tuple):
        ((src_input, src_len), (trg_label, trg_len)) = (src_tuple, trg_tuple)
        trg_input = tf.concat([[SOS_ID],trg_label[:-1]],axis=0)
        return ((src_input, src_len), (trg_input,trg_label, trg_len))

    dataset = dataset.map(MakeTrgInput)
    dataset = dataset.shuffle(10000)

    padded_shapes = (
        (tf.TensorShape([None]),tf.TensorShape([])),
        (tf.TensorShape([None]),tf.TensorShape([None]),tf.TensorShape([])))
    batched_dataset = dataset.padded_batch(batch_size,padded_shapes)
    return batched_dataset

class NMTModel(object):
    def __init__(self):
        self.enc_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE) for _ in range(NUM_LAYERS)])
        self.dec_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE) for _ in range(NUM_LAYERS)])

        self.src_embedding = tf.get_variable('src_emb',[SRC_VOCAB_SIZE,HIDDEN_SIZE])
        self.trg_embedding = tf.get_variable('trg_emb',[TRG_VOCAB_SIZE,HIDDEN_SIZE])

        if SHARE_EMB_AND_SOFTMAX:
            self.softmax_weight = tf.transpose(self.trg_embedding)
        else:
            self.softmax_weight = tf.get_variable('softmax_weight',[HIDDEN_SIZE,TRG_VOCAB_SIZE])

        self.softmax_bias = tf.get_variable('softmax_bias',[TRG_VOCAB_SIZE])

    def forward(self, src_input,src_size,trg_input,trg_label,trg_size):
         batch_size = tf.shape(src_input)[0]
         src_emb = tf.nn.embedding_lookup(self.src_embedding,src_input)
         trg_emb = tf.nn.embedding_lookup(self.trg_embedding,trg_input)

         src_emb = tf.nn.dropout(src_emb,KEEP_PROB)
         trg_emb = tf.nn.dropout(trg_emb,KEEP_PROB)

         with tf.variable_scope('encoder'):
             enc_outputs, enc_state = tf.nn.dynamic_rnn(self.enc_cell,src_emb,src_size,dtype=tf.float32)

         with tf.variable_scope('decoder'):
             dec_outputs, _ = tf.nn.dynamic_rnn(self.dec_cell,trg_emb,trg_size,initial_state=enc_state)

         output = tf.reshape(dec_outputs,[-1,HIDDEN_SIZE])
         logits = tf.matmul(output,self.softmax_weight)+self.softmax_bias
         loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(trg_label,[-1]),logits=logits)

         label_weights = tf.sequence_mask(trg_size,maxlen=tf.shape(trg_label)[1],dtype=tf.float32)
         label_weights = tf.reshape(label_weights,[-1])
         cost = tf.reduce_sum(label_weights*loss)
         cost_per_token = cost/tf.reduce_sum(label_weights)

         trainable_variables = tf.trainable_variables()
         grads = tf.gradients(cost/tf.to_float(batch_size),trainable_variables)
         grads,_ = tf.clip_by_global_norm(grads,MAX_GRAD_NORM)
         optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)
         train_op = optimizer.apply_gradients(zip(grads,trainable_variables))
         return cost_per_token,train_op

def run_epoch(session, cost_op, train_op, saver, step):
    while True:
        try:
            cost,_ = session.run([cost_op,train_op])
            if step%5 == 0:
                print("经过{0}次训练后，输出损失为{1}".format(step,cost))
            if step%200 == 0:
                saver.save(session,CHECKPOINT_PATH,global_step=step)
            step += 1
        except tf.errors.OutOfRangeError:
            break
    return step

def main():
    initializer = tf.random_uniform_initializer(-0.05,0.05)

    with tf.variable_scope('NMT_MODEL',reuse=None,initializer=initializer):
        train_model = NMTModel()

    data = MakeSrcTrgDataset(SRC_TRAIN_DATA,TRG_TRAIN_DATA,BATCH_SIZE)
    iterator = data.make_initializable_iterator()
    (src,src_size),(trg_input,trg_label,trg_size) = iterator.get_next()
    cost_op,train_op = train_model.forward(src,src_size,trg_input,trg_label,trg_size)
    saver = tf.train.Saver()
    step = 0
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(NUM_EPOCH):
            print("在第{0}次循环中".format(i+1))
            sess.run(iterator.initializer)
            step = run_epoch(sess,cost_op,train_op,saver,step)

if __name__ == '__main__':
    main()








