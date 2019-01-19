import tensorflow as tf
import numpy as np
from sklearn.metrics import f1_score
from scipy.special import softmax
import utils


class ModelConfig():
    """Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation. They can then call self.config.<hyperparameter_name> to
    get the hyperparameter settings.
    """

    def __init__(self, batch_size, max_seq_len):
        self.n_classes = 2
        self.max_gradient_norm = 1  # try with 5
        self.learning_rate = 1e-4
        self.embedding_size = 300
        self.rnn_hidden_size = 150
        self.dropout = 0
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.n_epochs = int((1306122 - 0.1*1306122) / self.batch_size)
        self.save_path = "saved/classifier.ckpt"


class SentenceClassifier():
    def __init__(self, config):
        self.config = config
        self.best_score = 0.0
        self.threshold = 0.5

    def build(self):
        tf.set_random_seed(1)
        self.add_placeholders()
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False,
                                       name='global_step')
        self.pred = self.add_prediction_op()
        self.loss = self.add_loss_op(self.pred)
        self.train_op = self.add_training_op(self.loss, self.global_step)
        self.batch_eval_metric, self.metric_update_op = self.add_eval_op(self.pred)

        self.add_summary_nodes()
        self.merged_summaries = tf.summary.merge_all()
        self.saver = tf.train.Saver()

    def add_placeholders(self):
        """Generates placeholder variables to represent the input tensors.

        Adds following nodes to the computational graph:

        input_placeholder: Input placeholder tensor of shape (None,
                           max_seq_len, embedding_size), type tf.float32
        labels_placeholder: Labels placeholder tensor of shape (None),
                            type tf.int32
        dropout_placeholder: Dropout value placeholder of shape (), i.e.
                             scalar, type tf.float32
        """
        self.input_placeholder = tf.placeholder(
            tf.float32,
            (None, self.config.max_seq_len, self.config.embedding_size),
            "input"
        )
        self.batch_seq_length_placeholder = tf.placeholder(tf.int32, (None, ),
                                                           "batch_seq_length")
        self.labels_placeholder = tf.placeholder(tf.int32, (None, ), "labels")
        self.config.dropout_placeholder = tf.placeholder(tf.float32, (), "dropout")

    def create_feed_dict(self, inputs_batch, batch_seq_length,
                         labels_batch=None, dropout=1):
        """Creates the feed_dict for training the given step.
        If label_batch is None, then no labels are added to feed_dict

        Args:
            inputs_batch: A batch of input data.
            labels_batch: A batch of label data.
            dropout: The dropout rate.

        Returns:
            feed_dict: The feed dictionary mapping from placeholders to values.
        """
        feed_dict = dict()
        feed_dict[self.input_placeholder] = inputs_batch
        feed_dict[self.batch_seq_length_placeholder] = batch_seq_length
        feed_dict[self.config.dropout_placeholder] = dropout
        if labels_batch is not None:
            feed_dict[self.labels_placeholder] = labels_batch

        return feed_dict

    def add_prediction_op(self):
        """Adds the core transformation for this model which transforms a batch
        of input data into a batch of predictions.

        Returns:
            pred: A tensor of shape (batch_size, n_classes)
        """
        pred = tf.get_variable(
            name='pred',
            shape=(self.config.batch_size, self.config.n_classes),
            initializer=tf.zeros_initializer()
        )

        return pred

    def add_loss_op(self, pred):
        """Adds loss ops to the computational graph.
        Args:
            pred: A tensor of shape (batch_size, n_classes)
        Returns:
            loss: A 0-d tensor (scalar)
        """
        loss = tf.Variable(0.0, 'loss')

        return loss

    def add_training_op(self, loss):
        """Creates an optimizer and applies the gradients to all trainable
        variables.
        Args:
            loss: Loss tensor, from cross_entropy_loss.
        Returns:
            train_op: The Op for training.
        """
        optimizer = None

        return optimizer

    def add_eval_op(self, pred):
        """Creates f1 evaluator of classifier"""
        f1_score, metric_update_op = tf.contrib.metrics.f1_score(
            self.labels_placeholder,
            tf.slice(tf.nn.softmax(self.pred), [0, 1], [-1, 1]),
            name='f1_score'
        )

        return f1_score, metric_update_op

    def add_summary_nodes(self):
        self.loss_summary = tf.summary.scalar("loss_summary", self.loss)
        self.eval_summary = tf.summary.scalar("f1_summary", self.metric_update_op)

    def fit(self, sess, inputs, seq_length, labels):
        feed_dict = self.create_feed_dict(inputs, seq_length, labels,
                                          self.config.dropout)

        loss, _, metric, summary = sess.run(
            [self.loss, self.train_op, self.metric_update_op,
             self.merged_summaries],
            feed_dict
        )

        return loss, metric, summary

    def evaluate(self, sess, data_gen):
        """Evaluates model on dev dataset and returns f1_score and predicted
        labels
        """
        pred_labels = np.array([], dtype=np.intp)
        labels = np.array([], dtype=np.intp)
        for inputs, seq_length, batch_labels in data_gen:
            feed_dict = self.create_feed_dict(inputs, seq_length, batch_labels,
                                              self.config.dropout)
            pred = sess.run(self.pred, feed_dict)
            pred = softmax(pred, -1)
            pred = utils.binarize(pred[:, 1], self.threshold)

            pred_labels = np.concatenate((pred_labels, pred))
            labels = np.concatenate((labels, batch_labels))

        score = f1_score(labels, pred_labels)

        return score, pred_labels

    def predict(self, sess, data_gen):
        """Predicts labels on unlabeled test set."""
        pred_labels = np.array([], dtype=np.intp)
        for inputs, seq_length in data_gen:
            feed_dict = self.create_feed_dict(inputs, seq_length,
                                              dropout=self.config.dropout)
            pred = sess.run(self.pred, feed_dict)
            pred = softmax(pred, -1)
            pred = utils.binarize(pred[:, 1], self.threshold)

            pred_labels = np.concatenate((pred_labels, pred))

        return pred_labels

    def save_best(self, sess, score):
        """Saves best model during train."""
        if score > self.best_score:
            self.best_score = score
            path_prefix = self.saver.save(sess, self.config.save_path,
                                          self.global_step)
            return path_prefix
        return "Skip saving"


class SentenceClassifierSeq2Seq(SentenceClassifier):
    def add_prediction_op(self):
        """Adds the core transformation for this model which transforms a batch
        of input data into a batch of predictions.

        Calculates forward pass of RNN on input sequence of length Tx:
        h_t = sigmoid(dot(W_hx, x_t) + dot(W_hh, h_(t-1) + b_t)
        After, calculates models prediction from last cell's activation h_Tx:
        h_drop = Dropout(h_Tx, dropout_rate)
        pred = dot(h_drop, W_ho) + b_o

        Returns:
            pred: A tensor of shape (batch_size, n_classes)
        """
        rnn_cell = tf.nn.rnn_cell.BasicRNNCell(self.config.rnn_hidden_size)
        rnn_cell_dropout = tf.nn.rnn_cell.DropoutWrapper(
            rnn_cell,
            input_keep_prob=0.9,
            output_keep_prob=1.0,
            state_keep_prob=0.6
        )
        #initial_state = rnn_cell.zero_state(batch_size = None, dtype=tf.float32)

        outputs, state = tf.nn.dynamic_rnn(
            cell=rnn_cell_dropout,
            inputs=self.input_placeholder,
            sequence_length=self.batch_seq_length_placeholder,
            dtype=tf.float32
            #initial_state=initial_state
        )

        h_drop = tf.nn.dropout(state, keep_prob=1)

        with tf.name_scope("classifier"):
            self.W_ho = tf.get_variable(
                "W_ho",
                (self.config.rnn_hidden_size, self.config.n_classes),
                tf.float32,
                tf.contrib.layers.xavier_initializer(),
                trainable=True
            )
            self.b_o = tf.get_variable(
                "bo",
                (1, self.config.n_classes),
                tf.float32, tf.zeros_initializer(),
                trainable=True
            )
            pred = tf.matmul(h_drop, self.W_ho) + self.b_o

        return pred

    def add_loss_op(self, pred):
        """Adds ops for the cross entropy loss to the computational graph.
        The loss is averaged over all examples in the current minibatch.

        Args:
            pred: A tensor of shape (batch_size, n_classes) containing the
                  output of the neural network before the softmax layer.
        Returns:
            loss: A 0-d tensor
        """
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.labels_placeholder,
            logits=pred,
            name="loss"
        )
        loss = tf.reduce_mean(loss)

        return loss

    def add_training_op(self, loss, global_step):
        """Creates an optimizer and applies the gradients to all trainable
        variables.
        Args:
            loss: Loss tensor, from cross_entropy_loss.
        Returns:
            train_op: The Op for training.
        """

        # Calculate and clip gradients
        params = tf.trainable_variables()
        gradients = tf.gradients(loss, params)
        clipped_gradients, _ = tf.clip_by_global_norm(
            gradients,
            self.config.max_gradient_norm
        )

        # Optimization
        optimizer = tf.train.AdamOptimizer(self.config.learning_rate)
        optimizer = optimizer.apply_gradients(zip(clipped_gradients, params),
                                              global_step, "adam_optimizer")

        return optimizer
