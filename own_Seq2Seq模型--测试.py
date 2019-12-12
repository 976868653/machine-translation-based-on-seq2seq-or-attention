import tensorflow as tf
import codecs
import sys

CHECKPOINT_PATH = './seq2seq_ckpt-9000'

HIDDEN_SIZE = 1024
NUM_LAYERS = 2
SRC_VOCAB_SIZE = 10000
TRG_VOCAB_SIZE = 4000
SHARE_EMB_AND_SOFTMAX = True

SRC_VOCAB = "./en.vocab"
TRG_VOCAB = "./zh.vocab"

SOS_ID = 1
EOS_ID = 2

class NMTModel(object):
    # 在模型的初始化函数中定义模型要用到的变量。
    def __init__(self):
        # 定义编码器和解码器所使用的LSTM结构。
        self.enc_cell = tf.nn.rnn_cell.MultiRNNCell(
          [tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE)
           for _ in range(NUM_LAYERS)])
        self.dec_cell = tf.nn.rnn_cell.MultiRNNCell(
          [tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE)
           for _ in range(NUM_LAYERS)])

        # 为源语言和目标语言分别定义词向量。
        self.src_embedding = tf.get_variable(
            "src_emb", [SRC_VOCAB_SIZE, HIDDEN_SIZE])
        self.trg_embedding = tf.get_variable(
            "trg_emb", [TRG_VOCAB_SIZE, HIDDEN_SIZE])

        # 定义softmax层的变量
        if SHARE_EMB_AND_SOFTMAX:
            self.softmax_weight = tf.transpose(self.trg_embedding)
        else:
            self.softmax_weight = tf.get_variable(
               "weight", [HIDDEN_SIZE, TRG_VOCAB_SIZE])
        self.softmax_bias = tf.get_variable(
            "softmax_bias", [TRG_VOCAB_SIZE])

    def inference(self, src_input):
        src_size = tf.convert_to_tensor([len(src_input)], dtype=tf.int32)
        src_input = tf.convert_to_tensor([src_input], dtype=tf.int32)
        src_emb = tf.nn.embedding_lookup(self.src_embedding, src_input)

        with tf.variable_scope('encoder'):
            enc_output, enc_state = tf.nn.dynamic_rnn(self.enc_cell,src_emb,src_size,dtype=tf.float32)


        MAX_DEC_LEN = 100

        with tf.variable_scope('decoder/rnn/multi_rnn_cell'):
            init_array = tf.TensorArray(dtype=tf.int32,size=0,dynamic_size=True,clear_after_read=False)
            init_array = init_array.write(0,SOS_ID)
            init_loop_var = (enc_state, init_array, 0)

            def continue_loop_condition(state, trg_ids, step):
                return tf.reduce_all(tf.logical_and(tf.not_equal(trg_ids.read(step),EOS_ID),tf.less(step,MAX_DEC_LEN-1)))

            def loop_body(state, trg_ids, step):
                trg_input = [trg_ids.read(step)]
                trg_emb = tf.nn.embedding_lookup(self.trg_embedding,
                                                 trg_input)
                dec_outputs, next_state = self.dec_cell.call(state=state,inputs=trg_emb)
                output = tf.reshape(dec_outputs,[-1,HIDDEN_SIZE])
                logits = tf.matmul(output,self.softmax_weight)+self.softmax_bias
                next_id = tf.argmax(logits,axis=1,output_type=tf.int32)

                trg_ids = trg_ids.write(step+1,next_id[0])
                return next_state,trg_ids,step+1

            state, trg_ids, step = tf.while_loop(
                continue_loop_condition, loop_body, init_loop_var)
            return trg_ids.stack()


def main():
    with tf.variable_scope('nmt_model',reuse=None):
        model = NMTModel()

    # 定义个测试句子。
    test_en_text = "I love you . <eos>"
    print(test_en_text)

    # 根据英文词汇表，将测试句子转为单词ID。
    with codecs.open(SRC_VOCAB, "r", "utf-8") as f_vocab:
        src_vocab = [w.strip() for w in f_vocab.readlines()]
        src_id_dict = dict((src_vocab[x], x) for x in range(len(src_vocab)))
    test_en_ids = [(src_id_dict[token] if token in src_id_dict else src_id_dict['<unk>'])
                   for token in test_en_text.split()]
    print(test_en_ids)
    # 建立解码所需的计算图。
    output_op = model.inference(test_en_ids)
    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess, CHECKPOINT_PATH)

    # 读取翻译结果。
    output_ids = sess.run(output_op)
    print(output_ids)

    with codecs.open(TRG_VOCAB, "r", "utf-8") as f_vocab:
        trg_vocab = [w.strip() for w in f_vocab.readlines()]
    output_text = ''.join([trg_vocab[x] for x in output_ids])

    # 输出翻译结果。
    print(output_text.encode('utf8').decode(sys.stdout.encoding))
    sess.close()

if __name__ == "__main__":
    main()

