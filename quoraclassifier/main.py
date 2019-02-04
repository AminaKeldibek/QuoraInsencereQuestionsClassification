import tensorflow as tf
import pandas as pd
import utils

from data_model import QuoraQuestionsModelStreamer, DataConfig
from sentence_classifier import SentenceClassifierConv
from sentence_classifier import SentenceClassifierSeq2SeqGRU
from sentence_classifier import ModelConfig, SentenceClassifierSeq2SeqGRUBinary


SAVE_EPOCH_STEP = 500
BATCH_SIZE = pow(2, 7)
MAX_SEQ_LEN = 70
EMBED_SIZE = 300


def train_model():
    data_model = QuoraQuestionsModelStreamer(DataConfig(), BATCH_SIZE, MAX_SEQ_LEN, EMBED_SIZE)
    classifier = SentenceClassifierSeq2SeqGRU(ModelConfig(), BATCH_SIZE, MAX_SEQ_LEN, EMBED_SIZE)
    train_gen = data_model.train_batch_generator()

    graph = tf.Graph()
    with graph.as_default():
        classifier.build()
        init = (tf.initializers.global_variables(),
                tf.initializers.local_variables())

    sess = tf.Session(graph=graph)
    writer = tf.summary.FileWriter("../tf_logdir", graph)

    sess.run(init)

    for i in range(classifier.n_epochs):
        inputs, seq_length, labels = next(train_gen)
        loss, metric, summary = classifier.fit(sess, inputs, seq_length,
                                               labels)
        writer.add_summary(summary, i)

        if i % SAVE_EPOCH_STEP == 0:
            print (f"Trained for {i} epochs")
            print (f"Train Loss/f1_score is {loss:.2f} / {metric:.2f}")

            score, _, _ = classifier.evaluate(
                sess,
                data_model.test_batch_generator(data_model.config.dev_dir),
                classifier.threshold
            )
            print (f"Dev f1_score is {score:.2f}")
            path_prefix = classifier.save_best(sess, score)

            print("Model saved in path: %s" % path_prefix)
    writer.close()
    sess.close()

    return classifier.best_model_path


def find_best_threshold(path_prefix):
    data_model = QuoraQuestionsModelStreamer(DataConfig(), BATCH_SIZE, MAX_SEQ_LEN, EMBED_SIZE)
    classifier = SentenceClassifierSeq2SeqGRU(ModelConfig(), BATCH_SIZE, MAX_SEQ_LEN, EMBED_SIZE)

    graph = tf.Graph()
    with graph.as_default():
        classifier.build()
        init = tf.initializers.global_variables()  # restore best graph vars

    sess = tf.Session(graph=graph)
    sess.run(init)
    classifier.saver.restore(sess, path_prefix)

    score, pred_labels, labels = classifier.evaluate(
        sess,
        data_model.test_batch_generator(data_model.config.train_dir)
    )

    best_threshold = utils.optimize_f1(pred_labels, labels)

    sess.close()

    return best_threshold


def test_model(path_prefix, threshold=0.5):
    data_model = QuoraQuestionsModelStreamer(DataConfig(), BATCH_SIZE, MAX_SEQ_LEN, EMBED_SIZE)
    classifier = SentenceClassifierSeq2SeqGRU(ModelConfig(), BATCH_SIZE, MAX_SEQ_LEN, EMBED_SIZE)

    graph = tf.Graph()
    with graph.as_default():
        classifier.build()
        init = tf.initializers.global_variables()  # restore best graph vars

    sess = tf.Session(graph=graph)
    sess.run(init)
    classifier.saver.restore(sess, path_prefix)

    score, pred_labels, labels = classifier.evaluate(
        sess,
        data_model.test_batch_generator(data_model.config.test_dir),
        threshold
    )
    print (f"Test f1_score is {score:.2f}")

    sess.close()

    return score, pred_labels, labels


def predict(path_prefix, threshold=0.5):
    data_model = QuoraQuestionsModelStreamer(DataConfig(), BATCH_SIZE, MAX_SEQ_LEN, EMBED_SIZE)
    classifier = SentenceClassifierSeq2SeqGRUBinary(ModelConfig(), BATCH_SIZE, MAX_SEQ_LEN, EMBED_SIZE)

    graph = tf.Graph()
    with graph.as_default():
        classifier.build()
        init = tf.initializers.global_variables()  # restore best graph vars

    sess = tf.Session(graph=graph)
    sess.run(init)
    classifier.saver.restore(sess, path_prefix)

    labels = classifier.predict(
        sess,
        data_model.predict_batch_generator(),
        threshold
    )
    sess.close()

    labels_df = pd.DataFrame(
        {'qid': data_model.get_predict_ids(), 'prediction': labels},
        columns=['qid', 'prediction']
    )
    labels_df.to_csv("../data/submission.csv", index=False)


if __name__ == '__main__':
    best_model_path = train_model()
    #optimal_treshold = find_best_threshold(best_model_path)
    #print ("Optimal threshold is", optimal_treshold)
    #best_model_path = "../saved_models/classifier.ckpt-7501"
    ##test_model(best_model_path)
    #predict("saved_models/classifier.ckpt-9001", optimal_treshold)
