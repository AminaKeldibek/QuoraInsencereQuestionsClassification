import pickle
import numpy as np
import pandas as pd
import nltk
import time


class DataConfig():
    pretrained_vectors_file = '../data/glove.840B.300d.txt'
    dict_file = '../data/processed/word2idx.txt'
    embedding_file = '../data/processed/embedding.npy'
    train_file = "../data/train.csv"
    parsed_train_file = "../data/processed/parsed_train.txt"

    embedding_size = 300
    max_seq_len = 50
    include_unknown = True
    unknown_token = "<UNK>"
    embedding_sample_size = 10000


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
        questions.bin: list of lists containing token indexes (integers)

        labels.bin: list of integers, corressponding class of quora question
        """
        labels = pd.read_csv(self.config.train_file, usecols=["target"])
        labels = list(labels.values[:, 0])
        questions = pd.read_csv("../data/train.csv", usecols=["question_text"],
                                index_col=False)
        self.load_dicts()
        unk_idx = self.word2idx[self.config.unknown_token]

        with open(self.config.parsed_train_file, 'w') as fo:
            for label, quest in zip(labels, questions.question_text):
                tokens = nltk.word_tokenize(quest.lower())
                if self.config.include_unknown:
                    idxs = [self.word2idx.get(token, unk_idx) for token in
                            tokens]
                else:
                    idxs = [self.word2idx.get(token) for token in tokens]
                    idxs = [idx for idx in idxs if idx]
                fo.write(str(" ".join(str(num) for num in idxs)) + " " +
                         str(label) + "\n")


class QuoraQuestionsModelStreamer(QuoraQuestionsModel):
    def __init__(self, data_config):
        self.config = data_config
        self.word2idx = dict()
        self.idx2word = dict()

    def sample_generator(self):
        """Yields sequence, sequence length, and label.
        """
        fi = open(self.config.parsed_train_file)
        for line in fi:
            list_line = line.split(" ")
            label = int(list_line[-1])
            sequence = np.array(list_line[:-1], dtype=np.intp)
            yield sequence, sequence.shape[0], label

    def batch_generator(self, batch_size):
        """Yields
        input: numpy 2D array of shape (batch_size, max_seq_len,
                                            embedding_size)
        labels: numpy 2D array of shape (batch_size, )
        sequence lengths: numpy 2D array of shape (batch_size, )
        """
        sample_gen = self.sample_generator()
        self.embedding = np.load(self.config.embedding_file)

        labels = np.zeros((batch_size), dtype=np.intp)
        seq_lengths = np.zeros((batch_size), dtype=np.intp)

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


def main_parser():
    data_model = QuoraQuestionsModelParser(DataConfig())
    #data_model.construct_dict()
    #data_model.construct_embedding()
    #data_model.add_unknown_token()
    #data_model.sentences_2_idxs()


def main_streamer():
    data_model = QuoraQuestionsModelStreamer(DataConfig())
    gen = data_model.batch_generator(2)

    for i in range(2):
        start = time.time()
        print (next(gen))
        end = time.time()
        print ("elapsed time" + str(end - start))


if __name__ == '__main__':
    main_parser()
    main_streamer()
