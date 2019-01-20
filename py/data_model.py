import pickle
import numpy as np
import pandas as pd
import nltk
import utils
import time
import random

random.seed(1)
np.random.seed(1)


class DataConfig():
    def __init__(self, batch_size, max_seq_len):
        self.pretrained_vectors_file = '../data/glove.840B.300d.txt'
        self.dict_file = '../data/processed/word2idx.txt'
        self.embedding_file = '../data/processed/embedding.npy'
        self.train_file = "../data/train.csv"
        self.parsed_train_file_pos = "../data/processed/parsed_train_pos.txt"
        self.parsed_train_file_neg = "../data/processed/parsed_train_neg.txt"

        self.train_dir = "../data/processed/train/"
        self.dev_dir = "../data/processed/dev/"
        self.test_dir = "../data/processed/test/"

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
        questions = pd.read_csv("../data/train.csv", usecols=["question_text"],
                                index_col=False)
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
        self.merge_pos_neg_helper(self.config.test_dir)
        self.merge_pos_neg_helper(self.config.dev_dir)


class QuoraQuestionsModelStreamer(QuoraQuestionsModel):
    def train_sample_generator(self, fi):
        """Yields sequence, sequence length from input file."""
        while True:
            line = fi.readline()
            if not line:
                fi.seek(0)
                continue
            sequence = np.array(line.split(" "), dtype=np.intp)
            yield sequence, sequence.shape[0]

    def labeled_sample_generator(self, fi):
        """Yields sequence, sequence length, and label from input file.

        Args:
            fi: file object opened for reading
        """
        for line in fi:
            line_list = line.split(" ")
            label = int(line_list[-1])
            sequence = np.array(line_list[:-1], dtype=np.intp)
            yield sequence, sequence.shape[0], label

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
        sample_gen = self.labeled_sample_generator(fi)
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


def main_parser():
    data_model = QuoraQuestionsModelParser(DataConfig())
    data_model.construct_dict()
    data_model.construct_embedding()
    data_model.add_unknown_token()
    data_model.sentences_2_idxs()
    data_model.split_train_test_dev()
    data_model.merge_pos_neg()


def main_streamer():
    data_model = QuoraQuestionsModelStreamer(DataConfig(2))
    gen = data_model.test_batch_generator(data_model.config.dev_dir)

    for input, length, label in gen:
        start = time.time()
        print ((input.shape, length, label))
        end = time.time()
        print ("elapsed time" + str(end - start))


if __name__ == '__main__':
    #main_parser()
    #main_streamer()
    pass
