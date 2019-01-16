import tensorflow as tf

from data_model import QuoraQuestionsModelStreamer, DataConfig
from sentence_classifier import SentenceClassifierSeq2Seq, ModelConfig


data_model = QuoraQuestionsModelStreamer(DataConfig())
classifier = SentenceClassifierSeq2Seq(ModelConfig())
batch_generator = data_model.train_batch_generator(classifier.config.batch_size)

graph = tf.Graph()

with graph.as_default():
    classifier.build()
    init = (tf.initializers.global_variables(),
            tf.initializers.local_variables())
    merged_summaries = tf.summary.merge_all()

sess = tf.Session(graph=graph)
sess.run(init)
writer = tf.summary.FileWriter("logdir", graph)

for i in range(classifier.config.n_epochs):
    inputs, seq_length, labels = next(batch_generator)
    feed_dict = classifier.create_feed_dict(inputs, seq_length, labels)
    loss, metric, _, summary = sess.run(
        [classifier.loss, classifier.metric_update_op, classifier.train_op,
         merged_summaries],
        feed_dict
    )
    writer.add_summary(summary, i)
    if i % 100 == 0:
        print ("Loss/f1_score after epoch #{0} is {1} / {2}".format(i, loss, metric))

writer.close()
sess.close()
