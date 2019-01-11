import pickle
import numpy as np


class QuoraQuestionsModel():
    def __init__(self):
        self.word2idx = dict()
        self.idx2word = dict()
        self.embedding_shape = None
        self.embedding = np.zeros(self.embedding_shape)

    def construct_dict(self, glove_dir):
        """ Constructs word2idx dict from Glove pretrained embeddings and writes
        it to word2idx.txt file in data directory of the project.
        """
        i = 0
        word2idx = dict()

        fi = open(glove_dir, 'r')
        fo = open('../data/processed/word2idx.txt', 'wb')

        for line in fi:
            word2idx[line.split(" ")[0]] = i
            i += 1

        pickle.Pickler(fo, 4).dump(word2idx)
        fi.close()
        fo.close()

    def load_dicts(self):
        # Opening our object dumpfile.
        f = open('../data/processed/word2idx.txt', 'rb')
        self.word2idx = pickle.Unpickler(f).load()
        self.idx2wod = {val: key for key, val in self.word2idx.items()}
        f.close()

    def construct_embedding(self):
        pass

    def get_embedding(self):
        pass


def main():
    data_model = QuoraQuestionsModel()
    data_model.construct_dict('../data/glove.840B.300d.txt')


if __name__ == '__main__':
    main()
