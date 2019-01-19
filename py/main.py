import tensorflow as tf

from data_model import QuoraQuestionsModelStreamer, DataConfig
from sentence_classifier import SentenceClassifierSeq2Seq, ModelConfig


SAVE_EPOCH_STEP = 500
BATCH_SIZE = pow(2, 7)
MAX_SEQ_LEN = 70


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


def test_model(path_prefix):
    data_model = QuoraQuestionsModelStreamer(DataConfig(BATCH_SIZE, MAX_SEQ_LEN))
    classifier = SentenceClassifierSeq2Seq(ModelConfig(BATCH_SIZE, MAX_SEQ_LEN))

    graph = tf.Graph()
    with graph.as_default():
        classifier.build()
        init = tf.initializers.global_variables()  # restore best graph vars

    sess = tf.Session(graph=graph)
    sess.run(init)
    classifier.saver.restore(sess, path_prefix)

    score, labels = classifier.evaluate(
        sess,
        data_model.test_batch_generator(data_model.config.test_dir)
    )
    print ("Test f1_score is {0}".format(score))

    sess.close()


def predict():
    pass


if __name__ == '__main__':
    train_model()
    #test_model("saved/classifier.ckpt-4501")
