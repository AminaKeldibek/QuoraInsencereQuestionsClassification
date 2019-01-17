import tensorflow as tf

from data_model import QuoraQuestionsModelStreamer, DataConfig
from sentence_classifier import SentenceClassifierSeq2Seq, ModelConfig


data_model = QuoraQuestionsModelStreamer(DataConfig())
classifier = SentenceClassifierSeq2Seq(ModelConfig())
batch_generator = data_model.test_batch_generator(classifier.config.batch_size)

graph = tf.Graph()

with graph.as_default():
    classifier.build()
    init = (tf.initializers.global_variables(),
            tf.initializers.local_variables())

sess = tf.Session(graph=graph)
sess.run(init)

for i in range(classifier.config.n_epochs):
    inputs, seq_length, labels = next(batch_generator)
    feed_dict = classifier.create_feed_dict(inputs, seq_length, labels)

    metric,  = sess.run(
        [classifier.metric_update_op],
        feed_dict
    )

sess.close()
