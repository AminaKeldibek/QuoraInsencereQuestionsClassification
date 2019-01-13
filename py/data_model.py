import pickle
import numpy as np
import pandas as pd
import nltk


class DataConfig():
    pretrained_vectors_file = '../data/glove.840B.300d.txt'
    dict_file = '../data/processed/word2idx.txt'
    embedding_file = '../data/processed/embedding.bin'
    train_file = "../data/train.csv"
    labels_file = "../data/processed/labels.bin"
    parsed_questions_file = "../data/processed/parsed_questions.bin"

    embedding_size = 300
    max_seq_len = 30
    include_unknown = True
    unknown_token = "<UNK>"


class QuoraQuestionsModel():
    def __init__(self, data_config):
        self.config = data_config
        self.word2idx = dict()
        self.idx2word = dict()

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
        self.embedding_shape = (max(self.word2idx.values()) + 1,
                                self.config.embedding_size)
        self.embedding = np.zeros(self.embedding_shape)

        fi = open(self.config.pretrained_vectors_file, 'r')
        fo = open(self.config.embedding_file, 'wb')

        for line in fi:
            word_vec = line.split(" ")[1:]
            self.embedding[i, :] = np.array(word_vec, dtype=np.float32)
            i += 1

        np.save(fo, self.embedding)
        fi.close()
        fo.close()

    def add_unknown_token(self):
        """Adds unknown token to word2idx dictionary and computes vector as an
        average of random sample as suggested by Pennington
        (https://groups.google.com/forum/#!searchin/globalvectors/unk|sort:date/globalvectors/9w8ZADXJclA/hRdn4prm-XUJ)
        """
        samples_size = 10

        with open(self.config.dict_file, 'rb') as fi:
            word2idx = pickle.Unpickler(fi).load()

        with open(self.config.dict_file, 'wb') as fi:
            word2idx[self.config.unknown_token] = max(word2idx.values()) + 1
            pickle.Pickler(fi, 4).dump(word2idx)

        embedding = np.load(self.config.embedding_file)
        sample_idxs = np.random.randint(0, embedding.shape[0], samples_size)
        unknown_vector = np.mean(embedding[sample_idxs, :], axis=0)
        embedding = np.vstack((embedding, unknown_vector))
        np.save(self.config.embedding_file, embedding)

    def sentences_2_idxs(self):
        """Replaces each Quora question with indexes corressponding to
        respective position of tokens in embedding matrix. If include_unknown
        is true, then replaces with corressponding index, ignores otherwise.

        Creates 2 binary files:
        questions.bin: list of lists containing token indexes (integers)

        labels.bin: list of integers, corressponding class of # QUESTION:
        """
        parsed_questions = []

        labels = pd.read_csv(self.config.train_file, usecols=["target"])
        labels = list(labels.values[:, 0])
        questions = pd.read_csv("../data/train.csv", usecols=["question_text"],
                                index_col=False)
        self.load_dicts()

        with open(self.config.labels_file, 'wb') as fo:
            pickle.Pickler(fo, 4).dump(labels)
        del labels

        unk_idx = self.word2idx[self.config.unknown_token]
        for quest in questions.question_text:
            tokens = nltk.word_tokenize(quest.lower())
            if self.config.include_unknown:
                idxs = [self.word2idx.get(token, unk_idx) for token in tokens]
            else:
                idxs = [self.word2idx.get(token) for token in tokens]
                idxs = [idx for idx in idxs if idx]
            parsed_questions.append(idxs)

        with open(self.config.parsed_questions_file, 'wb') as fo:
            pickle.Pickler(fo, 4).dump(parsed_questions)


class QuoraQuestionsModelStreamer(QuoraQuestionsModel):
    def __init__(self, data_config):
        self.config = data_config
        self.word2idx = dict()
        self.idx2word = dict()

    def sample_generator():
        """Yields sequence, sequence length, and label.
        """


    def batch_generator():
        """Yields
        batch input: numpy 2D array of shape (batch_size, max_seq_len,
                                            embedding_size)
        labels: numpy 2D array of shape (batch_size, )
        sequence lengths: numpy 2D array of shape (batch_size, )
        """


def main():
    data_model = QuoraQuestionsModel(DataConfig())
    #data_model.construct_dict()
    #data_model.construct_embedding()
    #data_model.add_unknown_token()
    #data_model.sentences_2_idxs()


if __name__ == '__main__':
    main()
