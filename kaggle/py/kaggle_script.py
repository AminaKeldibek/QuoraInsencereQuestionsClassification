import pickle
import numpy as np
import pandas as pd
import nltk
import random
import tensorflow as tf
import os
from sklearn.metrics import f1_score


random.seed(1)
np.random.seed(1)


SAVE_EPOCH_STEP = 500
BATCH_SIZE = pow(2, 7)
MAX_SEQ_LEN = 70


def create_dir(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise


def binarize(x, threshold):
    x_out = np.zeros_like(x)
    x_out[x > threshold] = 1
    x_out[x <= threshold] = 0

    return (x_out)


def logsumexp(a, axis=None, b=None, keepdims=False, return_sign=False):
    """Compute the log of the sum of exponentials of input elements."""
    if b is not None:
        a, b = np.broadcast_arrays(a, b)
        if np.any(b == 0):
            a = a + 0.  # promote to at least float
            a[b == 0] = -np.inf

    a_max = np.amax(a, axis=axis, keepdims=True)

    if a_max.ndim > 0:
        a_max[~np.isfinite(a_max)] = 0
    elif not np.isfinite(a_max):
        a_max = 0

    if b is not None:
        b = np.asarray(b)
        tmp = b * np.exp(a - a_max)
    else:
        tmp = np.exp(a - a_max)

    # suppress warnings about log of zero
    with np.errstate(divide='ignore'):
        s = np.sum(tmp, axis=axis, keepdims=keepdims)
        if return_sign:
            sgn = np.sign(s)
            s *= sgn  # /= makes more sense but we need zero -> zero
        out = np.log(s)

    if not keepdims:
        a_max = np.squeeze(a_max, axis=axis)
    out += a_max

    if return_sign:
        return out, sgn
    else:
        return out


def softmax(x, axis=None):
    """
    Softmax function
    """
    # compute in log space for numerical stability
    return np.exp(x - logsumexp(x, axis=axis, keepdims=True))


class DataConfig():
    def __init__(self, batch_size, max_seq_len):
        # Input file
        self.pretrained_vectors_file = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'
        self.train_file = "../input/train.csv"
        self.predict_file = "../input/test.csv"

        # Generated files
        self.dict_file = '../processed/word2idx.txt'
        self.embedding_file = '../processed/embedding.npy'
        self.parsed_train_file_pos = "../processed/parsed_train_pos.txt"
        self.parsed_train_file_neg = "../processed/parsed_train_neg.txt"
        self.parsed_predict_file = "../processed/parsed_predict.txt"

        self.train_dir = "../processed/train/"
        self.dev_dir = "../processed/dev/"
        self.test_dir = "../processed/test/"

        self.embedding_size = 300
        self.max_seq_len = max_seq_len
        self.include_unknown = True
        self.unknown_token = "<UNK>"
        self.embedding_sample_size = 10000

        self.dev_ratio = 0.05
        self.test_ratio = 0.05

        self.batch_size = batch_size
        self.class_probs = [0.9, 0.1]


class QuoraQuestionsModel():
    def __init__(self, data_config):
        create_dir("../processed")
        self.config = data_config
        self.word2idx = dict()
        self.idx2word = dict()
        self.embedding = None

    def load_dicts(self):
        f = open(self.config.dict_file, 'rb')
        self.word2idx = pickle.Unpickler(f).load()
        self.idx2wod = {val: key for key, val in self.word2idx.items()}
        f.close()

    def load_embedding(self):
        if self.embedding is None:
            self.embedding = np.load(self.config.embedding_file)

    def get_predict_ids(self):
        ids = pd.read_csv(self.config.predict_file, usecols=["qid"],
                          index_col=False)
        return ids.values[:, 0]


class QuoraQuestionsModelParser(QuoraQuestionsModel):
    def construct_dict(self):
        """ Constructs word2idx dict from Glove pretrained embeddings and writes
        it to word2idx.txt file in data directory of the project.
        """
        i = 0
        word2idx = dict()

        fi = open(self.config.pretrained_vectors_file, 'r')
        fo = open(self.config.dict_file, 'wb')

        for line in fi:
            word2idx[line.split(" ")[0]] = i
            i += 1

        pickle.Pickler(fo, 4).dump(word2idx)
        fi.close()
        fo.close()

    def construct_embedding(self):
        """ Creates embedding matrix from input file and writes to binary file
        Considering numpy row-major order stores word vector row-wise.
        """
        i = 0
        self.load_dicts()
        embedding_shape = (max(self.word2idx.values()) + 1,
                           self.config.embedding_size)
        embedding = np.zeros(embedding_shape)

        with open(self.config.pretrained_vectors_file, 'r') as fi:
            for line in fi:
                word_vec = line.split(" ")[1:]
                embedding[i, :] = np.array(word_vec, dtype=np.float32)
                i += 1

        np.save(self.config.embedding_file, embedding)

    def add_unknown_token(self):
        """Adds unknown token to word2idx dictionary and computes vector as an
        average of random sample as suggested by Pennington
        (https://groups.google.com/forum/#!searchin/globalvectors/unk|sort:date/globalvectors/9w8ZADXJclA/hRdn4prm-XUJ)
        """
        with open(self.config.dict_file, 'rb') as fi:
            word2idx = pickle.Unpickler(fi).load()

        with open(self.config.dict_file, 'wb') as fi:
            word2idx[self.config.unknown_token] = max(word2idx.values()) + 1
            pickle.Pickler(fi, 4).dump(word2idx)

        embedding = np.load(self.config.embedding_file)
        sample_idxs = np.random.randint(0, embedding.shape[0],
                                        self.config.embedding_sample_size)
        unknown_vector = np.mean(embedding[sample_idxs, :], axis=0)
        embedding = np.vstack((embedding, unknown_vector))
        np.save(self.config.embedding_file, embedding)

    def sentences_2_idxs(self):
        """Replaces each Quora question with indexes corressponding to
        respective position of tokens in embedding matrix. If include_unknown
        is true, then replaces with corressponding index, ignores otherwise.

        Creates 2 binary files:
        parsed_train_pos.txt: list of lists containing token indexes (integers)
                              of positive class

        parsed_train_neg.txt: list of lists containing token indexes (integers)
                              of negative class
        """
        fo_pos = open(self.config.parsed_train_file_pos, 'w')
        fo_neg = open(self.config.parsed_train_file_neg, 'w')

        labels = pd.read_csv(self.config.train_file, usecols=["target"])
        labels = list(labels.values[:, 0])
        questions = pd.read_csv(self.config.train_file,
                                usecols=["question_text"], index_col=False)
        self.load_dicts()
        unk_idx = self.word2idx[self.config.unknown_token]

        for label, quest in zip(labels, questions.question_text):
            tokens = nltk.word_tokenize(quest.lower())
            if self.config.include_unknown:
                idxs = [self.word2idx.get(token, unk_idx) for token in
                        tokens]
            else:
                idxs = [self.word2idx.get(token) for token in tokens]
                idxs = [idx for idx in idxs if idx]
            out_line = (str(" ".join(str(num) for num in idxs)) + "\n")
            if label == 1:
                fo_pos.write(out_line)
            else:
                fo_neg.write(out_line)

    def predict_sentences_2_idxs(self):
        """Replaces each Quora question with indexes corressponding to
        respective position of tokens in embedding matrix. If include_unknown
        is true, then replaces with corressponding index, ignores otherwise.

        Creates 2 binary files:
        parsed_train_pos.txt: list of lists containing token indexes (integers)
                              of positive class

        parsed_train_neg.txt: list of lists containing token indexes (integers)
                              of negative class
        """
        fo = open(self.config.parsed_predict_file, 'w')
        questions = pd.read_csv(self.config.predict_file,
                                usecols=["question_text"], index_col=False)
        self.load_dicts()
        unk_idx = self.word2idx[self.config.unknown_token]

        for quest in questions.question_text:
            tokens = nltk.word_tokenize(quest.lower())
            if self.config.include_unknown:
                idxs = [self.word2idx.get(token, unk_idx) for token in
                        tokens]
            else:
                idxs = [self.word2idx.get(token) for token in tokens]
                idxs = [idx for idx in idxs if idx]
            fo.write((str(" ".join(str(num) for num in idxs)) + "\n"))

    def split_helper(self, file_name, class_name):
        train_file, dev_file, test_file = map(
            lambda path: open(path + class_name + ".txt", "w"),
            (self.config.train_dir, self.config.dev_dir, self.config.test_dir)
        )

        with open(file_name) as fi:
            data = fi.readlines()

        train_size = (1 - self.config.dev_ratio - self.config.test_ratio)
        train_size = int(train_size * len(data))
        dev_size = int(self.config.dev_ratio * len(data))

        train_file.writelines(data[:train_size])
        dev_file.writelines(data[train_size: train_size + dev_size])
        test_file.writelines(data[train_size + dev_size:])

        map(lambda fo: fo.close(), (train_file, dev_file, test_file))

    def split_train_test_dev(self):
        """Splits data into train, dev, and test sets and saves these into
        separate directories.
        """
        for dir_name in (self.config.train_dir, self.config.dev_dir,
                         self.config.test_dir):
            create_dir(dir_name)

        self.split_helper(self.config.parsed_train_file_pos, 'pos')
        self.split_helper(self.config.parsed_train_file_neg, 'neg')

    def merge_pos_neg_helper(self, dir_name):
        pos = open(dir_name + "pos.txt").readlines()
        neg = open(dir_name + "neg.txt").readlines()

        pos = [seq.replace("\n", "") + ' 1\n' for seq in pos]
        neg = [seq.replace("\n", "") + ' 0\n' for seq in neg]

        out = pos + neg
        random.shuffle(out)

        with open(dir_name + "all.txt", "w") as fo:
            fo.writelines(out)

    def merge_pos_neg(self):
        self.merge_pos_neg_helper(self.config.test_dir)
        self.merge_pos_neg_helper(self.config.dev_dir)

    def parse_all(self):
        self.construct_dict()
        self.construct_embedding()
        self.add_unknown_token()
        self.sentences_2_idxs()
        self.predict_sentences_2_idxs()
        self.split_train_test_dev()
        self.merge_pos_neg()


class QuoraQuestionsModelStreamer(QuoraQuestionsModel):
    def train_sample_generator(self, fi):
        """Yields sequence, sequence length from input file.
        Reads file from beginning after reaching the end.
        """
        while True:
            line = fi.readline()
            if not line:
                fi.seek(0)
                continue
            sequence = np.array(line.split(" "), dtype=np.intp)
            yield sequence, sequence.shape[0]

    def dev_sample_generator(self, fi):
        """Yields sequence, sequence length, and label from input file.

        Args:
            fi: file object opened for reading
        """
        for line in fi:
            line_list = line.split(" ")
            label = int(line_list[-1])
            sequence = np.array(line_list[:-1], dtype=np.intp)
            yield sequence, sequence.shape[0], label

    def predict_sample_generator(self, fi):
        """Yields sequence, sequence length from input file.

        Args:
            fi: file object opened for reading
        """
        for line in fi:
            sequence = np.array(line.split(" "), dtype=np.intp)
            yield sequence, sequence.shape[0]

    def train_batch_generator(self):
        """Generates train batches by randomly selecting labels from both
        classes to avoid imbalance.
        Yields
        input: numpy 2D array of shape (batch_size, max_seq_len,
                                            embedding_size)
        labels: numpy 2D array of shape (batch_size, )
        sequence lengths: numpy 2D array of shape (batch_size, )
        """
        seq_lengths = np.zeros((self.config.batch_size), dtype=np.intp)
        fis = (self.config.train_dir + "pos.txt",
               self.config.train_dir + "neg.txt")
        fi_pos, fi_neg = map(open, fis)
        sample_gen_pos, sample_gen_neg = map(
            lambda fi: self.train_sample_generator(fi),
            (fi_pos, fi_neg)
        )
        self.load_embedding()

        while True:
            input = np.zeros((self.config.batch_size, self.config.max_seq_len,
                              self.config.embedding_size))
            labels = np.random.choice([0, 1], self.config.batch_size,
                                      p=self.config.class_probs)
            for i in range(self.config.batch_size):
                if labels[i] == 1:
                    sequence, seq_lengths[i] = next(sample_gen_pos)
                else:
                    sequence, seq_lengths[i] = next(sample_gen_neg)

                if seq_lengths[i] > self.config.max_seq_len:
                    seq_lengths[i] = self.config.max_seq_len
                    sequence = sequence[:seq_lengths[i]]
                input[i, 0:seq_lengths[i], :] = self.embedding[sequence, :]
            yield input, seq_lengths, labels

        map(lambda fi: fi.close(), (fi_pos, fi_neg))

    def test_batch_generator(self, dir_name):
        """Generates test batches from all.txt in test/ or dev/ directories
        where samples are shuffled and labeled.
        Yields
        input: numpy 2D array of shape (batch_size, max_seq_len,
                                            embedding_size)
        labels: numpy 2D array of shape (batch_size, )
        sequence lengths: numpy 2D array of shape (batch_size, )
        """
        input = np.zeros((self.config.batch_size, self.config.max_seq_len,
                          self.config.embedding_size))
        seq_lengths = np.zeros((self.config.batch_size), dtype=np.intp)
        labels = np.zeros((self.config.batch_size), dtype=np.intp)
        i = 0

        fi = open(dir_name + "all.txt")
        sample_gen = self.dev_sample_generator(fi)
        self.load_embedding()

        for sequence, seq_length, label in sample_gen:
            seq_lengths[i], labels[i] = seq_length, label
            if seq_lengths[i] > self.config.max_seq_len:
                seq_lengths[i] = self.config.max_seq_len
                sequence = sequence[:seq_lengths[i]]
            input[i, 0:seq_lengths[i], :] = self.embedding[sequence, :]

            i += 1

            if i == self.config.batch_size:
                yield input, seq_lengths, labels
                input = np.zeros(
                    (self.config.batch_size, self.config.max_seq_len,
                     self.config.embedding_size)
                )
                i = 0

        if i < self.config.batch_size:
            yield input[:i, :, :], seq_lengths[:i], labels[:i]

        fi.close()

    def predict_batch_generator(self):
        """Generates test batches from all.txt in test/ or dev/ directories
        where samples are shuffled and labeled.
        Yields
        input: numpy 2D array of shape (batch_size, max_seq_len,
                                            embedding_size)
        sequence lengths: numpy 2D array of shape (batch_size, )
        """
        input = np.zeros((self.config.batch_size, self.config.max_seq_len,
                          self.config.embedding_size))
        seq_lengths = np.zeros((self.config.batch_size), dtype=np.intp)
        i = 0

        fi = open(self.config.parsed_predict_file)
        sample_gen = self.predict_sample_generator(fi)
        self.load_embedding()

        for sequence, seq_length, in sample_gen:
            seq_lengths[i] = seq_length
            if seq_lengths[i] > self.config.max_seq_len:
                seq_lengths[i] = self.config.max_seq_len
                sequence = sequence[:seq_lengths[i]]
            input[i, 0:seq_lengths[i], :] = self.embedding[sequence, :]

            i += 1

            if i == self.config.batch_size:
                yield input, seq_lengths
                input = np.zeros(
                    (self.config.batch_size, self.config.max_seq_len,
                     self.config.embedding_size)
                )
                i = 0

        if i < self.config.batch_size:
            yield input[:i, :, :], seq_lengths[:i]

        fi.close()


class ModelConfig():
    """Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation. They can then call self.config.<hyperparameter_name> to
    get the hyperparameter settings.
    """

    def __init__(self, batch_size, max_seq_len):
        self.n_classes = 2
        self.max_gradient_norm = 1  # try with 5
        self.learning_rate = 5e-4
        self.embedding_size = 300
        self.rnn_hidden_size = 150
        self.dropout = 0
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.n_epochs = int((1306122 - 0.1*1306122) / self.batch_size)
        self.save_path = "saved/classifier.ckpt"


class SentenceClassifier():
    def __init__(self, config):
        self.config = config
        self.best_score = 0.0
        self.best_model_path = None
        self.threshold = 0.5

    def build(self):
        tf.set_random_seed(1)
        self.add_placeholders()
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False,
                                       name='global_step')
        self.pred = self.add_prediction_op()
        self.loss = self.add_loss_op(self.pred)
        self.train_op = self.add_training_op(self.loss, self.global_step)
        self.batch_eval_metric, self.metric_update_op = self.add_eval_op(self.pred)

        self.add_summary_nodes()
        self.merged_summaries = tf.summary.merge_all()
        self.saver = tf.train.Saver()

    def add_placeholders(self):
        """Generates placeholder variables to represent the input tensors.

        Adds following nodes to the computational graph:

        input_placeholder: Input placeholder tensor of shape (None,
                           max_seq_len, embedding_size), type tf.float32
        labels_placeholder: Labels placeholder tensor of shape (None),
                            type tf.int32
        dropout_placeholder: Dropout value placeholder of shape (), i.e.
                             scalar, type tf.float32
        """
        self.input_placeholder = tf.placeholder(
            tf.float32,
            (None, self.config.max_seq_len, self.config.embedding_size),
            "input"
        )
        self.batch_seq_length_placeholder = tf.placeholder(tf.int32, (None, ),
                                                           "batch_seq_length")
        self.labels_placeholder = tf.placeholder(tf.int32, (None, ), "labels")
        self.config.dropout_placeholder = tf.placeholder(tf.float32, (), "dropout")

    def create_feed_dict(self, inputs_batch, batch_seq_length,
                         labels_batch=None, dropout=1):
        """Creates the feed_dict for training the given step.
        If label_batch is None, then no labels are added to feed_dict

        Args:
            inputs_batch: A batch of input data.
            labels_batch: A batch of label data.
            dropout: The dropout rate.

        Returns:
            feed_dict: The feed dictionary mapping from placeholders to values.
        """
        feed_dict = dict()
        feed_dict[self.input_placeholder] = inputs_batch
        feed_dict[self.batch_seq_length_placeholder] = batch_seq_length
        feed_dict[self.config.dropout_placeholder] = dropout
        if labels_batch is not None:
            feed_dict[self.labels_placeholder] = labels_batch

        return feed_dict

    def add_prediction_op(self):
        """Adds the core transformation for this model which transforms a batch
        of input data into a batch of predictions.

        Returns:
            pred: A tensor of shape (batch_size, n_classes)
        """
        pred = tf.get_variable(
            name='pred',
            shape=(self.config.batch_size, self.config.n_classes),
            initializer=tf.zeros_initializer()
        )

        return pred

    def add_loss_op(self, pred):
        """Adds loss ops to the computational graph.
        Args:
            pred: A tensor of shape (batch_size, n_classes)
        Returns:
            loss: A 0-d tensor (scalar)
        """
        loss = tf.Variable(0.0, 'loss')

        return loss

    def add_training_op(self, loss):
        """Creates an optimizer and applies the gradients to all trainable
        variables.
        Args:
            loss: Loss tensor, from cross_entropy_loss.
        Returns:
            train_op: The Op for training.
        """
        optimizer = None

        return optimizer

    def add_eval_op(self, pred):
        """Creates f1 evaluator of classifier"""
        f1_score, metric_update_op = tf.contrib.metrics.f1_score(
            self.labels_placeholder,
            tf.slice(tf.nn.softmax(self.pred), [0, 1], [-1, 1]),
            name='f1_score'
        )

        return f1_score, metric_update_op

    def add_summary_nodes(self):
        self.loss_summary = tf.summary.scalar("loss_summary", self.loss)
        self.eval_summary = tf.summary.scalar("f1_summary", self.metric_update_op)

    def fit(self, sess, inputs, seq_length, labels):
        feed_dict = self.create_feed_dict(inputs, seq_length, labels,
                                          self.config.dropout)

        loss, _, metric, summary = sess.run(
            [self.loss, self.train_op, self.metric_update_op,
             self.merged_summaries],
            feed_dict
        )

        return loss, metric, summary

    def evaluate(self, sess, data_gen):
        """Evaluates model on dev dataset and returns f1_score and predicted
        labels
        """
        pred_labels = np.array([], dtype=np.intp)
        labels = np.array([], dtype=np.intp)
        for inputs, seq_length, batch_labels in data_gen:
            feed_dict = self.create_feed_dict(inputs, seq_length, batch_labels,
                                              self.config.dropout)
            pred = sess.run(self.pred, feed_dict)
            pred = softmax(pred, -1)
            pred = binarize(pred[:, 1], self.threshold)

            pred_labels = np.concatenate((pred_labels, pred))
            labels = np.concatenate((labels, batch_labels))

        score = f1_score(labels, pred_labels)

        return score, pred_labels

    def predict(self, sess, data_gen):
        """Predicts labels on unlabeled test set."""
        pred_labels = np.array([], dtype=np.intp)
        for inputs, seq_length in data_gen:
            feed_dict = self.create_feed_dict(inputs, seq_length,
                                              dropout=self.config.dropout)
            pred = sess.run(self.pred, feed_dict)
            pred = softmax(pred, -1)
            pred = binarize(pred[:, 1], self.threshold)

            pred_labels = np.concatenate((pred_labels, pred))

        return pred_labels.astype(np.int8)

    def save_best(self, sess, score):
        """Saves best model during train."""
        if score > self.best_score:
            self.best_score = score
            path_prefix = self.saver.save(sess, self.config.save_path,
                                          self.global_step)
            self.best_model_path = path_prefix
            return path_prefix
        return "Skip saving"


class SentenceClassifierSeq2Seq(SentenceClassifier):
    def add_prediction_op(self):
        """Adds the core transformation for this model which transforms a batch
        of input data into a batch of predictions.

        Calculates forward pass of RNN on input sequence of length Tx:
        h_t = sigmoid(dot(W_hx, x_t) + dot(W_hh, h_(t-1) + b_t)
        After, calculates models prediction from last cell's activation h_Tx:
        h_drop = Dropout(h_Tx, dropout_rate)
        pred = dot(h_drop, W_ho) + b_o

        Returns:
            pred: A tensor of shape (batch_size, n_classes)
        """
        #rnn_cell = tf.nn.rnn_cell.BasicRNNCell(self.config.rnn_hidden_size)

        '''rnn_cell = tf.nn.rnn_cell.GRUCell(
            self.config.rnn_hidden_size,
            #activation=None,
            #reuse=None,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            bias_initializer=tf.zeros_initializer(),
            name="gru",
            dtype=tf.float32
        )'''

        rnn_cell = tf.nn.rnn_cell.LSTMCell(
            num_units=self.config.rnn_hidden_size,
            use_peepholes=False,
            #cell_clip=None,
            initializer=tf.contrib.layers.xavier_initializer(),
            num_proj=None,
            #proj_clip=None,
            #forget_bias=1.0,
            state_is_tuple=True,
            #activation=None,
            #reuse=None,
            name="lstm",
            dtype=tf.float32
        )

        '''rnn_cell_dropout = tf.nn.rnn_cell.DropoutWrapper(
            rnn_cell,
            input_keep_prob=1.0,
            output_keep_prob=1.0,
            state_keep_prob=0.8
        )'''
        #initial_state = rnn_cell.zero_state(batch_size = None, dtype=tf.float32)

        outputs, state = tf.nn.dynamic_rnn(
            cell=rnn_cell,
            inputs=self.input_placeholder,
            sequence_length=self.batch_seq_length_placeholder,
            dtype=tf.float32
            #initial_state=initial_state
        )
        state = tf.reshape(tf.slice(state, [0, 0, 0], [1, -1, -1]), (-1, self.config.rnn_hidden_size))
        h_drop = tf.nn.dropout(state, keep_prob=1.0)

        with tf.name_scope("classifier"):
            self.W_ho = tf.get_variable(
                "W_ho",
                (self.config.rnn_hidden_size, self.config.n_classes),
                tf.float32,
                tf.contrib.layers.xavier_initializer(),
                trainable=True
            )
            self.b_o = tf.get_variable(
                "bo",
                (1, self.config.n_classes),
                tf.float32, tf.zeros_initializer(),
                trainable=True
            )
            pred = tf.matmul(h_drop, self.W_ho) + self.b_o

        return pred

    def add_loss_op(self, pred):
        """Adds ops for the cross entropy loss to the computational graph.
        The loss is averaged over all examples in the current minibatch.

        Args:
            pred: A tensor of shape (batch_size, n_classes) containing the
                  output of the neural network before the softmax layer.
        Returns:
            loss: A 0-d tensor
        """
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.labels_placeholder,
            logits=pred,
            name="loss"
        )
        loss = tf.reduce_mean(loss)

        return loss

    def add_training_op(self, loss, global_step):
        """Creates an optimizer and applies the gradients to all trainable
        variables.
        Args:
            loss: Loss tensor, from cross_entropy_loss.
        Returns:
            train_op: The Op for training.
        """

        # Calculate and clip gradients
        params = tf.trainable_variables()
        gradients = tf.gradients(loss, params)
        clipped_gradients, _ = tf.clip_by_global_norm(
            gradients,
            self.config.max_gradient_norm
        )

        # Optimization
        optimizer = tf.train.AdamOptimizer(self.config.learning_rate)
        optimizer = optimizer.apply_gradients(zip(clipped_gradients, params),
                                              global_step, "adam_optimizer")

        return optimizer


# Main
def train_model():
    data_model = QuoraQuestionsModelStreamer(DataConfig(BATCH_SIZE, MAX_SEQ_LEN))
    classifier = SentenceClassifierSeq2Seq(ModelConfig(BATCH_SIZE, MAX_SEQ_LEN))
    train_gen = data_model.train_batch_generator()

    graph = tf.Graph()
    with graph.as_default():
        classifier.build()
        init = (tf.initializers.global_variables(),
                tf.initializers.local_variables())

    sess = tf.Session(graph=graph)
    writer = tf.summary.FileWriter("logdir", graph)

    sess.run(init)

    for i in range(classifier.config.n_epochs):
        inputs, seq_length, labels = next(train_gen)
        loss, metric, summary = classifier.fit(sess, inputs, seq_length,
                                               labels)
        writer.add_summary(summary, i)

        if i % SAVE_EPOCH_STEP == 0:
            print (f"Trained for {i} epochs")
            print (f"Train Loss/f1_score is {loss:.2f} / {metric:.2f}")

            score, labels = classifier.evaluate(
                sess,
                data_model.test_batch_generator(data_model.config.dev_dir)
            )
            print (f"Dev f1_score is {score:.2f}")
            path_prefix = classifier.save_best(sess, score)

            print("Model saved in path: %s" % path_prefix)
    writer.close()
    sess.close()

    return classifier.best_model_path


def predict(path_prefix):
    data_model = QuoraQuestionsModelStreamer(DataConfig(BATCH_SIZE, MAX_SEQ_LEN))
    classifier = SentenceClassifierSeq2Seq(ModelConfig(BATCH_SIZE, MAX_SEQ_LEN))

    graph = tf.Graph()
    with graph.as_default():
        classifier.build()
        init = tf.initializers.global_variables()  # restore best graph vars

    sess = tf.Session(graph=graph)
    sess.run(init)
    classifier.saver.restore(sess, path_prefix)

    labels = classifier.predict(sess, data_model.predict_batch_generator())
    print(labels)
    sess.close()

    labels_df = pd.DataFrame(
        {'qid': data_model.get_predict_ids(), 'prediction': labels},
        columns=['qid', 'prediction']
    )
    labels_df.to_csv("submission.csv", index=False)


if __name__ == '__main__':
    QuoraQuestionsModelParser(DataConfig(BATCH_SIZE, MAX_SEQ_LEN)).parse_all()
    best_model_path = train_model()
    predict(best_model_path)
