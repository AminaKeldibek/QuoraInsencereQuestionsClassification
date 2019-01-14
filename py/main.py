import tensorflow as tf
import time

from data_model import QuoraQuestionsModelStreamer, DataConfig
from sentence_classifier import SentenceClassifierSeq2Seq, ModelConfig


data_model = QuoraQuestionsModelStreamer(DataConfig())
classifier = SentenceClassifierSeq2Seq(ModelConfig())

batch_generator = data_model.batch_generator(classifier.config.batch_size)

classifier_graph = tf.Graph()

with classifier_graph.as_default():
    classifier.build()
    init = tf.initializers.global_variables()

sess = tf.Session(graph=classifier_graph)
sess.run(init)

for i in range(classifier.config.n_epochs):
    start = time.time()
    inputs, seq_length, labels = next(batch_generator)
    feed_dict = classifier.create_feed_dict(inputs, seq_length, labels)
    loss, _ = sess.run([classifier.loss, classifier.train_op], feed_dict)
    print ("Loss after epoch #" + str(i) + "is " + str(loss))
    end = time.time()
    print ("Elapsed time for one epoch is " + str(end - start))

sess.close()
