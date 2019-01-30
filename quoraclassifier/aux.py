import pandas as pd
import pickle

from data_model import QuoraQuestionsModel, DataConfig
from collections import defaultdict
from utils import preprocess_text


def count_missing_words():
    data_model = QuoraQuestionsModel(DataConfig(), 10, 10, 300)
    data_model.load_dicts()
    questions = pd.read_csv(data_model.config.train_file,
                            usecols=["question_text"], index_col=False)
    word_count = defaultdict(int)
    for quest in questions.question_text:
        tokens = preprocess_text(quest)
        for token in tokens:
            if token not in data_model.word2idx:
                word_count[token] += 1

    fo = open("../data/missing_word_counts_new.txt", "wb")
    pickle.Pickler(fo, 4).dump(word_count)
    fo.close()


if __name__ == '__main__':
    count_missing_words()
