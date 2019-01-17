import tensorflow as tf

from data_model import QuoraQuestionsModelStreamer, DataConfig
from sentence_classifier import SentenceClassifierSeq2Seq, ModelConfig


SAVE_EPOCH_STEP = 100
BATCH_SIZE = pow(2, 7)


data_model = QuoraQuestionsModelStreamer(DataConfig(BATCH_SIZE))
classifier = SentenceClassifierSeq2Seq(ModelConfig(BATCH_SIZE))
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
    loss, metric, summary = classifier.fit(sess, inputs, seq_length, labels)
    writer.add_summary(summary, i)

    if i % SAVE_EPOCH_STEP == 0:
        print ("Loss/f1_score after epoch #{0} is {1} / {2}".format(i, loss,
                                                                    metric))
        score, labels = classifier.evaluate(
            data_model.test_batch_generator(data_model.config.dev_dir)
        )
        path_prefix = classifier.save_best(sess, score, labels)

print("Model saved in path: %s" % path_prefix)
writer.close()
sess.close()
