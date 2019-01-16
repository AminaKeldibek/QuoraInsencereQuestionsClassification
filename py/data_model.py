import pickle
import numpy as np
import pandas as pd
import nltk
import utils

import random
random.seed(1)

import time


class DataConfig():
    pretrained_vectors_file = '../data/glove.840B.300d.txt'
    dict_file = '../data/processed/word2idx.txt'
    embedding_file = '../data/processed/embedding.npy'
    train_file = "../data/train.csv"
    parsed_train_file_pos = "../data/processed/parsed_train_pos.txt"
    parsed_train_file_neg = "../data/processed/parsed_train_neg.txt"

    train_dir = "../data/processed/train/"
    dev_dir = "../data/processed/dev/"
    test_dir = "../data/processed/test/"

    embedding_size = 300
    max_seq_len = 50
    include_unknown = True
    unknown_token = "<UNK>"
    embedding_sample_size = 10000

    dev_ratio = 0.05
    test_ratio = 0.05


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
        parsed_train_pos.txt: list of lists containing token indexes (integers) of positive class

        parsed_train_neg.txt: list of lists containing token indexes (integers) of negative class
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
    def __init__(self, data_config):
        self.config = data_config
        self.word2idx = dict()
        self.idx2word = dict()

    def sample_generator(self, fi):
        """Yields sequence, sequence length from input file."""
        while True:
            line = fi.readline()
            if not line:
                fi.seek(0)
                continue
            sequence = np.array(line.split(" "), dtype=np.intp)
            yield sequence, sequence.shape[0]

    def labeled_sample_generator(self, fi):
        """Yields sequence, sequence length, and label from input file."""
        while True:
            line = fi.readline()
            if not line:
                fi.seek(0)
                continue
            line_list = line.split(" ")
            label = int(line_list[-1])
            sequence = np.array(line_list[:-1], dtype=np.intp)
            yield sequence, sequence.shape[0], label

    def train_batch_generator(self, batch_size):
        """Generates train batches by randomly selecting labels from both
        classes to avoid imbalance.
        Yields
        input: numpy 2D array of shape (batch_size, max_seq_len,
                                            embedding_size)
        labels: numpy 2D array of shape (batch_size, )
        sequence lengths: numpy 2D array of shape (batch_size, )
        """
        fis = (self.config.train_dir + "pos.txt",
               self.config.train_dir + "neg.txt")
        fi_pos, fi_neg = map(open, fis)
        sample_gen_pos, sample_gen_neg = map(
            lambda fi: self.sample_generator(fi),
            (fi_pos, fi_neg)
        )

        self.embedding = np.load(self.config.embedding_file)
        seq_lengths = np.zeros((batch_size), dtype=np.intp)

        while True:
            input = np.zeros((batch_size, self.config.max_seq_len,
                              self.config.embedding_size))
            labels = np.random.randint(0, 2, batch_size, np.intp)
            for i in range(batch_size):
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

    def test_batch_generator(self, batch_size, dir_name):
        """Generates test batches from all.txt in test/ or dev/ directories
        where samples are shuffled and labeled.
        Yields
        input: numpy 2D array of shape (batch_size, max_seq_len,
                                            embedding_size)
        labels: numpy 2D array of shape (batch_size, )
        sequence lengths: numpy 2D array of shape (batch_size, )
        """
        fi = open(dir_name + "all.txt")
        sample_gen = self.labeled_sample_generator(fi)

        self.embedding = np.load(self.config.embedding_file)
        seq_lengths = np.zeros((batch_size), dtype=np.intp)
        labels = np.zeros((batch_size), dtype=np.intp)

        while True:
            input = np.zeros((batch_size, self.config.max_seq_len,
                              self.config.embedding_size))
            for i in range(batch_size):
                sequence, seq_lengths[i], labels[i] = next(sample_gen)
                if seq_lengths[i] > self.config.max_seq_len:
                    seq_lengths[i] = self.config.max_seq_len
                    sequence = sequence[:seq_lengths[i]]
                input[i, 0:seq_lengths[i], :] = self.embedding[sequence, :]
            yield input, seq_lengths, labels

        fi.close()


def main_parser():
    #data_model = QuoraQuestionsModelParser(DataConfig())
    #data_model.construct_dict()
    #data_model.construct_embedding()
    #data_model.add_unknown_token()
    #data_model.sentences_2_idxs()
    #data_model.split_train_test_dev()
    #data_model.merge_pos_neg()
    pass


def main_streamer():
    data_model = QuoraQuestionsModelStreamer(DataConfig())
    gen = data_model.test_batch_generator(4, data_model.config.dev_dir)

    for i in range(2):
        start = time.time()
        print (next(gen))
        end = time.time()
        print ("elapsed time" + str(end - start))


if __name__ == '__main__':
    main_parser()
    main_streamer()
