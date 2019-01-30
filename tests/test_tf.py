import tensorflow as tf
import numpy as np

from data_model import QuoraQuestionsModelStreamer, DataConfig
from sentence_classifier import SentenceClassifierConv, ModelConfig


SAVE_EPOCH_STEP = 500
BATCH_SIZE = pow(2, 7)
MAX_SEQ_LEN = 70
EMBED_SIZE = 300

data_model = QuoraQuestionsModelStreamer(DataConfig(), BATCH_SIZE, MAX_SEQ_LEN, EMBED_SIZE)
classifier = SentenceClassifierConv(ModelConfig(), BATCH_SIZE, MAX_SEQ_LEN, EMBED_SIZE)
train_gen = data_model.train_batch_generator()
inputs, seq_length, labels = next(train_gen)

graph = tf.Graph()
with graph.as_default():
    tf.set_random_seed(1)
    classifier.add_placeholders()
    classifier.pred = classifier.add_prediction_op()
    init = (tf.initializers.global_variables(),
            tf.initializers.local_variables())

sess = tf.Session(graph=graph)
writer = tf.summary.FileWriter("testlogdir", graph)

sess.run(init)

feed_dict = classifier.create_feed_dict(inputs, seq_length, labels, classifier.config.dropout)
pred = sess.run(classifier.pred, feed_dict)

print (pred.shape)

writer.close()
sess.close()


###
