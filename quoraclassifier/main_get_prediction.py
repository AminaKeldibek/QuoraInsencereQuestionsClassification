import tensorflow as tf

from data_model import DataConfig, OneInputModel
from sentence_classifier import SentenceClassifierSeq2SeqExtFeats
from sentence_classifier import ModelConfig


BATCH_SIZE = pow(2, 7)
MAX_SEQ_LEN = 90
EMBED_SIZE = 300


def predict(path_prefix, classifier, input_data, threshold=0.5):
    graph = tf.Graph()
    with graph.as_default():
        classifier.build()
        init = tf.initializers.global_variables()  # restore best graph vars

    sess = tf.Session(graph=graph)
    sess.run(init)
    classifier.saver.restore(sess, path_prefix)

    labels = classifier.predict_one_sample(
        sess,
        input_data,
        threshold
    )
    sess.close()
    print (labels)


def main():
    input_text = "I love sunny weather."
    input_text = "I hate Trump"
    optimal_treshold = 0.6
    best_model_path = "../saved_models/classifier.ckpt-4001"

    data_model = OneInputModel(
        DataConfig(),
        BATCH_SIZE, MAX_SEQ_LEN, EMBED_SIZE
    )
    input_data = data_model.preprocess_text_input(input_text)
    sequence, seq_length, unique_count = input_data

    classifier = SentenceClassifierSeq2SeqExtFeats(
        ModelConfig(),
        BATCH_SIZE, MAX_SEQ_LEN, EMBED_SIZE
    )

    predict(best_model_path, classifier, input_data, optimal_treshold)


if __name__ == '__main__':
    main()
    #import sys

    #main(sys.argv[1:])
