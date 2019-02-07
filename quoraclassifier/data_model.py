import pickle
import numpy as np
import pandas as pd
#from quoraclassifier import utils
import utils
import time
import random

random.seed(1)
np.random.seed(1)


class DataConfig():
    # Input file
    word_vec_fi_glove = '../data/glove.840B.300d.txt'
    word_vec_fi_paragram = '../data/paragram_300_sl999.txt' # 1703756 words
    train_file = "../data/train.csv"
    predict_file = "../data/test.csv"

    # Generated files
    dict_file = '../data/processed/word2idx.txt'
    embedding_file = '../data/processed/embedding.npy'
    parsed_train_file_pos = "../data/processed/parsed_train_pos.txt"
    parsed_train_file_neg = "../data/processed/parsed_train_neg.txt"
    parsed_predict_file = "../data/processed/parsed_predict.txt"

    train_dir = "../data/processed/train/"
    dev_dir = "../data/processed/dev/"
    test_dir = "../data/processed/test/"

    include_unknown = True
    unknown_token = "<UNK>"
    embedding_sample_size = 10000

    dev_ratio = 0.05
    test_ratio = 0.05

    class_probs = [0.9, 0.1]


class QuoraQuestionsModel():
    def __init__(self, data_config, batch_size, max_seq_len, embedding_size):
        self.config = data_config
        self.word2idx = dict()
        self.idx2word = dict()
        self.embedding = None
        self.embedding_size = embedding_size
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.vocab_size = 0

    def load_dicts(self):
        with open(self.config.dict_file, 'rb') as f:
            self.word2idx = pickle.Unpickler(f).load()
        self.idx2word = {val: key for key, val in self.word2idx.items()}
        self.vocab_size = max(self.word2idx.values()) + 1

    def load_embedding(self):
        if self.embedding is None:
            self.embedding = np.load(self.config.embedding_file)
            self.vocab_size = self.embedding.shape[0]

    def load_all(self):
        self.load_dicts()
        self.load_embedding()

    def get_predict_ids(self):
        ids = pd.read_csv(self.config.predict_file, usecols=["qid"],
                          index_col=False)
        return ids.values[:, 0]

    def write_dict(self):
        with open(self.config.dict_file, 'wb') as fo:
            pickle.Pickler(fo, 4).dump(self.word2idx)

    def write_embedding(self):
        np.save(self.config.embedding_file, self.embedding)

    def write_all(self):
        self.write_dict()
        self.write_embedding()

    def clear_all(self):
        self.embedding = None
        self.word2idx = None
        self.idx2word = None
        self.vocab_size = 0


class QuoraQuestionsModelParser(QuoraQuestionsModel):
    def construct_dict(self):
        """ Constructs word2idx dict from Glove pretrained embeddings and writes
        it to word2idx.txt file in data directory of the project.
        """
        i = 0
        self.word2idx = dict()
        fi = open(self.config.word_vec_fi_glove, 'r')

        for line in fi:
            self.word2idx[line.split(" ")[0]] = i
            i += 1

        self.vocab_size = i
        self.write_dict()
        fi.close()

    def construct_embedding(self):
        """ Creates embedding matrix from input file and writes to binary file
        Considering numpy row-major order stores word vector row-wise.
        """
        i = 0
        self.load_dicts()
        embedding_shape = (max(self.word2idx.values()) + 1,
                           self.embedding_size)
        self.embedding = np.zeros(embedding_shape)

        with open(self.config.word_vec_fi_glove, 'r') as fi:
            for line in fi:
                word_vec = line.split(" ")[1:]
                self.embedding[i, :] = np.array(word_vec, dtype=np.float32)
                i += 1

        self.write_embedding()

    def embeddings_file_gen(self, fi):
        ''' Yields token and embedding from fi.

        Args:
            fi: file object opened for reading

        Yields:
            token: string
            embedding: numpy array of shape (1, self.embedding_size)
        '''
        for line in fi:
            line_list = line.split(" ")
            token = line_list[0]
            embedding = np.array(line_list[1:], dtype=np.float32)
            embedding = np.reshape(embedding, (1, -1))

            yield token, embedding

    def add_embedding(self, token, embedding):
        """Helper function to add new token to self.word2idx and embedding to
        self.embeddings"""
        self.word2idx[token] = self.vocab_size
        self.vocab_size += 1

        self.embedding = np.vstack((self.embedding, embedding))

    def add_paragram(self):
        """Averages word vectors that occur in both glove and paragram and
        creates union of two embeddings"""
        num_new_words = 720000
        new_embeddings = np.empty((num_new_words, self.embedding_size))
        concat_emb = np.zeros((2, self.embedding_size))
        new_word2idx = dict()
        new_words_count = 0

        self.load_all()
        fi = open(self.config.word_vec_fi_paragram, "r", encoding="utf8",
                  errors='ignore')
        embed_gen = self.embeddings_file_gen(fi)

        for token, embedding in embed_gen:
            if token not in self.word2idx:
                new_word2idx[token] = self.vocab_size
                new_embeddings[new_words_count, :] = embedding
                self.vocab_size += 1
                new_words_count += 1
            else:
                concat_emb[0, :] = self.embedding[self.word2idx[token]]
                concat_emb[1, :] = embedding
                self.embedding[self.word2idx[token]] = np.mean(
                    concat_emb,
                    axis=0
                )

        self.word2idx.update(new_word2idx)
        self.embedding = np.vstack((
            self.embedding,
            new_embeddings[:new_words_count, :]
        ))

        self.write_all()
        fi.close()

    def add_unknown_token(self):
        """Adds unknown token to word2idx dictionary and computes vector as an
        average of random sample as suggested by Pennington
        (https://groups.google.com/forum/#!searchin/globalvectors/unk|sort:date/globalvectors/9w8ZADXJclA/hRdn4prm-XUJ)
        """
        self.load_all()
        sample_idxs = np.random.randint(0, self.embedding.shape[0],
                                        self.config.embedding_sample_size)
        unknown_vector = np.mean(self.embedding[sample_idxs, :], axis=0)

        self.add_embedding(self.config.unknown_token, unknown_vector)
        self.write_all()

    def add_unk_to_dict(self, tokens):
        for token in tokens:
            if token not in self.word2idx:
                self.word2idx[token] = self.vocab_size
                self.vocab_size += 1
                self.num_unknown_words += 1

    def init_unknown_embeddings(self):
        oov_random = utils.xavier_weight_init((self.num_unknown_words, self.embedding_size))
        self.embedding = np.vstack((self.embedding, oov_random))

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
        self.load_dicts()
        labels = pd.read_csv(self.config.train_file, usecols=["target"])

        labels = list(labels.values[:, 0])
        questions = pd.read_csv(self.config.train_file,
                                usecols=["question_text"], index_col=False)
        unk_idx = self.word2idx.get(self.config.unknown_token)

        for label, quest in zip(labels, questions.question_text):
            tokens = utils.preprocess_text(quest)

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
        self.load_dicts()

        questions = pd.read_csv(self.config.predict_file,
                                usecols=["question_text"], index_col=False)
        unk_idx = self.word2idx[self.config.unknown_token]

        for quest in questions.question_text:
            tokens = utils.preprocess_text(quest)
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
            utils.create_dir(dir_name)

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
        self.merge_pos_neg_helper(self.config.train_dir)
        self.merge_pos_neg_helper(self.config.test_dir)
        self.merge_pos_neg_helper(self.config.dev_dir)

    def parse_all(self):
        #self.construct_dict()
        #self.construct_embedding()
        #self.load_all()
        #self.add_unknown_token()

        #self.add_paragram()

        self.sentences_2_idxs()
        self.predict_sentences_2_idxs()
        self.split_train_test_dev()
        self.merge_pos_neg()

class QuoraQuestionsModelStreamer(QuoraQuestionsModel):
    def train_sample_generator(self, fi):
        """Yields sequence, sequence length, count of unique tokens from input
        file. Reads file from beginning after reaching the end.
        """
        while True:
            line = fi.readline()
            if not line:
                fi.seek(0)
                continue
            sequence = np.array(line.split(" "), dtype=np.intp)
            yield sequence, sequence.shape[0], np.unique(sequence).shape[0]

    def dev_sample_generator(self, fi):
        """Yields sequence, sequence length, and label from input file.

        Args:
            fi: file object opened for reading
        """
        for line in fi:
            line_list = line.split(" ")
            label = int(line_list[-1])
            sequence = np.array(line_list[:-1], dtype=np.intp)
            yield sequence, sequence.shape[0], np.unique(sequence).shape[0], label

    def predict_sample_generator(self, fi):
        """Yields sequence, sequence length from input file.

        Args:
            fi: file object opened for reading
        """
        for line in fi:
            sequence = np.array(line.split(" "), dtype=np.intp)
            yield sequence, sequence.shape[0], np.unique(sequence).shape[0]

    def train_batch_generator(self):
        """Generates train batches by randomly selecting labels from both
        classes to avoid imbalance.
        Yields
        input: numpy 2D array of shape (batch_size, max_seq_len,
                                            embedding_size)
        sequence lengths: numpy 2D array of shape (batch_size, )
        unique counts: numpy 2D array of shape (batch_size, )
        labels: numpy 2D array of shape (batch_size, )
        """
        seq_lengths = np.zeros((self.batch_size), dtype=np.intp)
        unique_count = np.zeros((self.batch_size), dtype=np.intp)
        fis = (self.config.train_dir + "pos.txt",
               self.config.train_dir + "neg.txt")
        fi_pos, fi_neg = map(open, fis)
        sample_gen_pos, sample_gen_neg = map(
            lambda fi: self.train_sample_generator(fi),
            (fi_pos, fi_neg)
        )
        self.load_embedding()

        while True:
            input = np.zeros((self.batch_size, self.max_seq_len,
                              self.embedding_size))
            labels = np.random.choice([0, 1], self.batch_size,
                                      p=self.config.class_probs)
            for i in range(self.batch_size):
                if labels[i] == 1:
                    sequence, seq_lengths[i], unique_count[i] = next(sample_gen_pos)
                else:
                    sequence, seq_lengths[i], unique_count[i] = next(sample_gen_neg)

                if seq_lengths[i] > self.max_seq_len:
                    seq_lengths[i] = self.max_seq_len
                    sequence = sequence[:seq_lengths[i]]
                input[i, 0:seq_lengths[i], :] = self.embedding[sequence, :]
            yield input, seq_lengths, unique_count, labels

        map(lambda fi: fi.close(), (fi_pos, fi_neg))

    def test_batch_generator(self, dir_name):
        """Generates test batches from all.txt in train/ test/ or dev/ directories
        where samples are shuffled and labeled.
        Yields
        input: numpy 2D array of shape (batch_size, max_seq_len,
                                            embedding_size)
        labels: numpy 2D array of shape (batch_size, )
        sequence lengths: numpy 2D array of shape (batch_size, )
        unique counts: numpy 2D array of shape (batch_size, )
        """
        input = np.zeros((self.batch_size, self.max_seq_len,
                          self.embedding_size))
        seq_lengths = np.zeros((self.batch_size), dtype=np.intp)
        unique_counts = np.zeros((self.batch_size), dtype=np.intp)
        labels = np.zeros((self.batch_size), dtype=np.intp)
        i = 0

        fi = open(dir_name + "all.txt")
        sample_gen = self.dev_sample_generator(fi)
        self.load_embedding()

        for sequence, seq_length, unique_count, label in sample_gen:
            seq_lengths[i], labels[i], unique_counts[i] = seq_length, label, unique_count
            if seq_lengths[i] > self.max_seq_len:
                seq_lengths[i] = self.max_seq_len
                sequence = sequence[:seq_lengths[i]]
            input[i, 0:seq_lengths[i], :] = self.embedding[sequence, :]

            i += 1

            if i == self.batch_size:
                yield input, seq_lengths, unique_counts, labels
                input = np.zeros(
                    (self.batch_size, self.max_seq_len,
                     self.embedding_size)
                )
                i = 0

        if i < self.batch_size:
            yield input[:i, :, :], seq_lengths[:i], unique_counts[:i], labels[:i]

        fi.close()

    def predict_batch_generator(self):
        """Generates test batches from all.txt in test/ or dev/ directories
        where samples are shuffled and labeled.
        Yields
        input: numpy 2D array of shape (batch_size, max_seq_len,
                                            embedding_size)
        sequence lengths: numpy 2D array of shape (batch_size, )
        """
        input = np.zeros((self.batch_size, self.max_seq_len,
                          self.embedding_size))
        seq_lengths = np.zeros((self.batch_size), dtype=np.intp)
        unique_counts = np.zeros((self.batch_size), dtype=np.intp)
        i = 0

        fi = open(self.config.parsed_predict_file)
        sample_gen = self.predict_sample_generator(fi)
        self.load_embedding()

        for sequence, seq_length, unique_count in sample_gen:
            seq_lengths[i], unique_counts[i] = seq_length, unique_count
            if seq_lengths[i] > self.max_seq_len:
                seq_lengths[i] = self.max_seq_len
                sequence = sequence[:seq_lengths[i]]
            input[i, 0:seq_lengths[i], :] = self.embedding[sequence, :]

            i += 1

            if i == self.batch_size:
                yield input, seq_lengths, unique_counts
                input = np.zeros(
                    (self.batch_size, self.max_seq_len,
                     self.embedding_size)
                )
                i = 0

        if i < self.batch_size:
            yield input[:i, :, :], seq_lengths[:i], unique_counts[:i]

        fi.close()


def main_parser():
    data_model = QuoraQuestionsModelParser(DataConfig(), 100, 70, 300)
    data_model.parse_all()


def main_streamer():
    data_model = QuoraQuestionsModelStreamer(DataConfig(2, 70))
    gen = data_model.test_batch_generator(data_model.config.dev_dir)

    for input, length, label in gen:
        start = time.time()
        print ((input.shape, length, label))
        end = time.time()
        print ("elapsed time" + str(end - start))


if __name__ == '__main__':
    main_parser()
    #main_streamer()
