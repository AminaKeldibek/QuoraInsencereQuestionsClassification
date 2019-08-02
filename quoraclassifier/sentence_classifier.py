import tensorflow as tf
import numpy as np
from sklearn.metrics import f1_score
from scipy.special import softmax
import utils
from utils import sigmoid


class ModelConfig():
    """Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation. They can then call sef.config.<hyperparameter_name> to
    get the hyperparameter settings.
    """
    n_classes = 2
    max_gradient_norm = 5  # try with 1
    learning_rate = 1e-3
    rnn_hidden_size = 300
    dropout = 0
    save_path = "../saved_models/classifier.ckpt"


class SentenceClassifier():
    def __init__(self, config, batch_size, max_seq_len, embedding_size):
        self.config = config
        self.best_score = 0.0
        self.best_model_path = None
        self.threshold = 0.5
        self.embedding_size = embedding_size
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.n_epochs = int((1306122 - 0.1*1306122) / self.batch_size)

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
            (None, self.max_seq_len, self.embedding_size),
            "input"
        )
        self.batch_seq_length_placeholder = tf.placeholder(tf.int32, (None, ),
                                                           "batch_seq_length")
        self.batch_unique_count_placeholder = tf.placeholder(tf.int32, (None, ),
                                                           "batch_unique_count")
        self.labels_placeholder = tf.placeholder(tf.int32, (None, ), "labels")

        self.dropout_placeholder = tf.placeholder(tf.float32, (), "dropout")
        self.lr_placeholder = tf.placeholder(tf.float32, (), "lr")

    def create_feed_dict(self, inputs_batch, batch_seq_length,
                         batch_unique_count, labels_batch=None, dropout=1,
                         lr=None):
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
        feed_dict[self.batch_unique_count_placeholder] = batch_unique_count
        feed_dict[self.dropout_placeholder] = dropout
        if labels_batch is not None:
            feed_dict[self.labels_placeholder] = labels_batch
        if lr is not None:
            feed_dict[self.lr_placeholder] = lr
        return feed_dict

    def add_prediction_op(self):
        """Adds the core transformation for this model which transforms a batch
        of input data into a batch of predictions.

        Returns:
            pred: A tensor of shape (batch_size, n_classes)
        """
        pred = tf.get_variable(
            name='pred',
            shape=(self.batch_size, self.config.n_classes),
            initializer=tf.zeros_initializer()
        )

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

    def fit(self, sess, inputs, seq_length, unique_count, labels):
        feed_dict = self.create_feed_dict(inputs, seq_length, unique_count,
                                          labels, self.config.dropout)

        loss, _, metric, summary = sess.run(
            [self.loss, self.train_op, self.metric_update_op,
             self.merged_summaries],
            feed_dict
        )

        return loss, metric, summary

    def evaluate(self, sess, data_gen, threshold=None):
        """Evaluates model on dev dataset and returns f1_score and predicted
        labels. If threshold is provided, it binarizes predictions, otherwise
        returns softmax output.
        """
        score = None
        pred_labels = np.array([], dtype=np.intp)
        labels = np.array([], dtype=np.intp)
        for inputs, seq_length, unique_count, batch_labels in data_gen:
            feed_dict = self.create_feed_dict(inputs, seq_length, unique_count,
                                              batch_labels, self.config.dropout)
            pred = sess.run(self.pred, feed_dict)
            pred = softmax(pred, -1)[:, 1]
            if threshold is not None:
                pred = utils.binarize(pred, threshold)

            pred_labels = np.concatenate((pred_labels, pred))
            labels = np.concatenate((labels, batch_labels))

        if threshold is not None:
            score = f1_score(labels, pred_labels)

        return score, pred_labels, labels

    def predict(self, sess, data_gen, threshold=None):
        """Predicts labels on unlabeled test set. If threshold is provided,
        it binarizes predictions, otherwise returns softmax output.
        """
        pred_labels = np.array([], dtype=np.intp)
        for inputs, seq_length, unique_count in data_gen:
            feed_dict = self.create_feed_dict(inputs, seq_length, unique_count,
                                              dropout=self.config.dropout)
            pred = sess.run(self.pred, feed_dict)
            pred = softmax(pred, -1)[:, 1]
            if threshold is not None:
                pred = utils.binarize(pred, self.threshold)

            pred_labels = np.concatenate((pred_labels, pred))

        return pred_labels.astype(np.int8)

    def predict_one_sample(self, sess, input_data, threshold=None):
        pred_labels = np.array([], dtype=np.intp)
        sequence, seq_length, unique_count = input_data

        feed_dict = self.create_feed_dict(sequence, seq_length, unique_count,
                                          dropout=self.config.dropout)
        pred = sess.run(self.pred, feed_dict)
        pred = softmax(pred, -1)[:, 1]
        if threshold is not None:
            pred = utils.binarize(pred, self.threshold)

        pred_labels = np.concatenate((pred_labels, pred))

        return pred_labels.astype(np.int8)


    def save_best(self, sess, score):
        """Saves best model during train."""
        if score > self.best_score:
            self.best_score = score
            path_prefix = self.saver.save(sess, self.config.save_path,
                                          self.global_step)
            self.best_model_path = path_prefix
            return path_prefix
        return "Skip saving"


class SentenceClassifierSeq2SeqGRU(SentenceClassifier):
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

        x_dropout = tf.keras.layers.SpatialDropout1D(0.4).apply(self.input_placeholder)

        rnn_cell = tf.nn.rnn_cell.GRUCell(
            self.config.rnn_hidden_size,
            activation='relu',
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            bias_initializer=tf.zeros_initializer(),
            name="gru",
            dtype=tf.float32
        )

        rnn_cell_dropout = tf.nn.rnn_cell.DropoutWrapper(
            rnn_cell,
            input_keep_prob=0.9,
            output_keep_prob=1.0,
            state_keep_prob=0.8
        )

        outputs, state = tf.nn.dynamic_rnn(
            cell=rnn_cell,
            inputs=x_dropout,
            sequence_length=self.batch_seq_length_placeholder,
            dtype=tf.float32
            #initial_state=initial_state
        )

        h_drop = tf.nn.dropout(state, keep_prob=0.8)

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


class SentenceClassifierSeq2SeqExtFeats(SentenceClassifier):
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
            (None, self.max_seq_len, self.embedding_size),
            "input"
        )
        self.batch_seq_length_placeholder = tf.placeholder(tf.int32, (None, ),
                                                           "batch_seq_length")
        self.batch_unique_count_placeholder = tf.placeholder(tf.float32, (None, ),
                                                           "batch_unique_count")
        self.labels_placeholder = tf.placeholder(tf.int32, (None, ), "labels")

        self.dropout_placeholder = tf.placeholder(tf.float32, (), "dropout")
        self.lr_placeholder = tf.placeholder(tf.float32, (), "dropout")

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
        optimizer = tf.train.AdamOptimizer(self.lr_placeholder)
        optimizer = optimizer.apply_gradients(zip(clipped_gradients, params),
                                              global_step, "adam_optimizer")

        return optimizer

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

        #x_dropout = tf.keras.layers.SpatialDropout1D(0.4).apply(self.input_placeholder)
        layer_1_size = 150
        layer_2_size = 25
        num_aux_feats = 4

        rnn_cell_layer_1_fwd = tf.nn.rnn_cell.GRUCell(
            layer_1_size,
            activation='relu',
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            bias_initializer=tf.zeros_initializer(),
            name="gru_1",
            dtype=tf.float32
        )

        rnn_cell_layer_1_bwd = tf.nn.rnn_cell.GRUCell(
            layer_1_size,
            activation='relu',
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            bias_initializer=tf.zeros_initializer(),
            name="gru_1",
            dtype=tf.float32
        )

        rnn_cell_layer_2_fwd = tf.nn.rnn_cell.GRUCell(
            layer_2_size,
            activation='relu',
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            bias_initializer=tf.zeros_initializer(),
            name="gru_2",
            dtype=tf.float32
        )

        rnn_cell_layer_2_bwd = tf.nn.rnn_cell.GRUCell(
            layer_2_size,
            activation='relu',
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            bias_initializer=tf.zeros_initializer(),
            name="gru_2",
            dtype=tf.float32
        )

        outputs_1, states_1 = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=rnn_cell_layer_1_fwd,
            cell_bw=rnn_cell_layer_1_bwd,
            inputs=self.input_placeholder,
            sequence_length=self.batch_seq_length_placeholder,
            dtype=tf.float32
        )
        h1 = tf.concat(outputs_1, axis=2, name="concat1")

        outputs_2, states_2 = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=rnn_cell_layer_2_fwd,
            cell_bw=rnn_cell_layer_2_bwd,
            inputs=h1,
            dtype=tf.float32
        )
        h2 = tf.concat(states_2, axis=1, name="concat2")

        h3 = tf.concat(
            [
                h2,
                tf.reshape(self.batch_unique_count_placeholder, (-1, 1)),
                tf.reduce_max(h2, axis=1, keepdims=True)
            ],
            axis=1,
            name="concat3"
        )
        num_aux_feats = 2

        #h_drop = tf.nn.dropout(state, keep_prob=0.8)

        with tf.name_scope("classifier"):
            self.W_ho = tf.get_variable(
                "W_ho",
                (layer_2_size * 2 + num_aux_feats, self.config.n_classes),
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
            pred = tf.matmul(h3, self.W_ho) + self.b_o

        return pred

    def fit(self, sess, inputs, seq_length, unique_count, labels, lr):
        feed_dict = self.create_feed_dict(inputs, seq_length, unique_count,
                                          labels, self.config.dropout, lr)

        loss, _, metric, summary = sess.run(
            [self.loss, self.train_op, self.metric_update_op,
             self.merged_summaries],
            feed_dict
        )

        return loss, metric, summary

    def evaluate(self, sess, data_gen, threshold=None):
        """Evaluates model on dev dataset and returns f1_score and predicted
        labels. If threshold is provided, it binarizes predictions, otherwise
        returns softmax output.
        """
        score = None
        pred_labels = np.array([], dtype=np.intp)
        labels = np.array([], dtype=np.intp)
        for inputs, seq_length, unique_count, batch_labels in data_gen:
            feed_dict = self.create_feed_dict(inputs, seq_length, unique_count,
                                              batch_labels, self.config.dropout)
            pred = sess.run(self.pred, feed_dict)
            pred = softmax(pred, -1)[:, 1]
            if threshold is not None:
                pred = utils.binarize(pred, threshold)

            pred_labels = np.concatenate((pred_labels, pred))
            labels = np.concatenate((labels, batch_labels))

        if threshold is not None:
            score = f1_score(labels, pred_labels)

        return score, pred_labels, labels


class SentenceClassifierSeq2SeqLSTM(SentenceClassifier):
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
        x_dropout = tf.keras.layers.SpatialDropout1D(0.4).apply(self.input_placeholder)

        rnn_cell = tf.nn.rnn_cell.LSTMCell(
            num_units=self.config.rnn_hidden_size,
            use_peepholes=False,
            #cell_clip=None,
            initializer=tf.contrib.layers.xavier_initializer(),
            num_proj=None,
            #proj_clip=None,
            #forget_bias=1.0,
            state_is_tuple=True,
            #activation=None,
            #reuse=None,
            name="lstm",
            dtype=tf.float32
        )

        '''rnn_cell_dropout = tf.nn.rnn_cell.DropoutWrapper(
            rnn_cell,
            input_keep_prob=1.0,
            output_keep_prob=1.0,
            state_keep_prob=0.8
        )'''

        outputs, state = tf.nn.dynamic_rnn(
            cell=rnn_cell,
            inputs=x_dropout,
            sequence_length=self.batch_seq_length_placeholder,
            dtype=tf.float32
        )
        state = tf.reshape(
            tf.slice(state, [0, 0, 0], [1, -1, -1]),
            (-1, self.config.rnn_hidden_size)
        )

        h_drop = tf.nn.dropout(state, keep_prob=1.0)
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


class SentenceClassifierSeq2SeqAttention(SentenceClassifier):
    def add_prediction_op(self):
        """Adds the core transformation for this model which transforms a batch
        of input data into a batch of predictions.

        Calculates bidirectional forward pass on input sequence of length Tx using GRU Cell:
        h_t = sigmoid(dot(W_hx, x_t) + dot(W_hh, h_(t-1) + b_h)# change equation here

        This produces cell outputs as a tuple of forward and backward rnn cells.
        The shape of this matrix is (None, T_x, n_h).

        Cell outputs are then inputted to dense layer with sofmtax activatio to
        calculate weights alpha:
        alpha =  softmax(A * W_alpha_a), alpha has shape (None, T_x, 1)

        Using alpha and outputs context vector is calculated as:
        context = reduce_sum(output o alpha, axis=1)

        Finally, class is predicted as
        h_drop = Dropout(context, dropout_rate)
        pred = dot(h_drop, W_ho) + b_o

        Returns:
            pred: A tensor of shape (batch_size, n_classes)
        """
        rnn_cell_fwd = tf.nn.rnn_cell.GRUCell(
            self.config.rnn_hidden_size,
            activation='relu',
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            bias_initializer=tf.zeros_initializer(),
            name="gru",
            dtype=tf.float32
        )

        rnn_cell_fwd_dropout = tf.nn.rnn_cell.DropoutWrapper(
            rnn_cell_fwd,
            input_keep_prob=1.0,
            output_keep_prob=1.0,
            state_keep_prob=0.9,
            #variational_recurrent=False,
            #input_size=None,
            dtype=tf.float32,
            seed=1,
        )

        rnn_cell_bwd = tf.nn.rnn_cell.GRUCell(
            self.config.rnn_hidden_size,
            activation='relu',
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            bias_initializer=tf.zeros_initializer(),
            name="gru",
            dtype=tf.float32
        )

        rnn_cell_bwd_dropout = tf.nn.rnn_cell.DropoutWrapper(
            rnn_cell_bwd,
            input_keep_prob=1.0,
            output_keep_prob=1.0,
            state_keep_prob=0.9,
            #variational_recurrent=False,
            #input_size=None,
            dtype=tf.float32,
            seed=1,
        )

        outputs, states = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=rnn_cell_fwd_dropout,
            cell_bw=rnn_cell_bwd_dropout,
            inputs=self.input_placeholder,
            sequence_length=self.batch_seq_length_placeholder,
            dtype=tf.float32
        )
        states = tf.concat(outputs, axis=2)

        # Attention here
        self.W_alpha_a = tf.get_variable(
            "W_alpha_a",
            (self.config.rnn_hidden_size * 2, 1),
            tf.float32,
            tf.contrib.layers.xavier_initializer(),
            trainable=True
        )

        states_prime = tf.reshape(states, shape=(-1, self.config.rnn_hidden_size * 2))
        alpha_prime = tf.linalg.matmul(states_prime, self.W_alpha_a)
        alpha = tf.reshape(alpha_prime, shape=(-1, self.max_seq_len, 1))
        alpha = tf.nn.softmax(alpha, axis=1)
        context = tf.reduce_sum(tf.multiply(states, alpha), axis=1)

        '''h_dropout = tf.nn.dropout(
            context,
            keep_prob=0.8,
            seed=1,
        )'''

        with tf.name_scope("classifier"):
            self.W_ho = tf.get_variable(
                "W_ho",
                (self.config.rnn_hidden_size * 2, self.config.n_classes),
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
            pred = tf.matmul(context, self.W_ho) + self.b_o

        return pred


class SentenceClassifierConv(SentenceClassifier):
    def __init__(self, config, batch_size, max_seq_len, embedding_size):
        self.config = config
        self.best_score = 0.0
        self.best_model_path = None
        self.threshold = 0.5
        self.embedding_size = embedding_size
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.n_epochs = int((1306122 - 0.1*1306122) / self.batch_size)

        self.in_channels = 1
        self.out_channels = 100
        self.ngrams = [2, 3, 4]
        self.num_filters = len(self.ngrams) * self.out_channels
        self.data_format = 'NHWC'
        self.conv_strides = [1, 1, 1, 1]
        self.max_pool_strides = [1, 1, 1, 1]

    def get_filter(self, ngram, out_channels):
        """Creates filter for convolution of shape
        (ngram, embed_size, in_channel, out_channels).
        If init is None, initialized with xavier_initializer
        """
        filter = tf.get_variable(
            "filter",
            (ngram, self.embedding_size, self.in_channels, out_channels),
            tf.float32,
            tf.contrib.layers.xavier_initializer(),
            trainable=True
        )
        return filter

    def add_prediction_op(self):
        """Uses Kim's conv network architecture for sentence classification.
            feature map has shape of (batch_size, out_height, out_width, out_channels)

        Returns:
            pred: A tensor of shape (batch_size, n_classes)
        """
        max_pool_out_list = []

        input = tf.reshape(
            self.input_placeholder,
            (-1, self.max_seq_len, self.embedding_size, 1)
        )
        with tf.name_scope("convolution"):
            for ngram in self.ngrams:
                with tf.variable_scope(str(ngram) + "_gram_conv"):
                    filters = self.get_filter(ngram, self.out_channels)
                    b_conv = tf.get_variable(
                        "b_conv",
                        shape=(self.out_channels),
                        initializer=tf.zeros_initializer()
                    )
                    z = tf.nn.conv2d(
                        input,
                        filters,
                        strides=self.conv_strides,
                        padding="VALID",
                        use_cudnn_on_gpu=True,
                        data_format=self.data_format,
                        #dilations=[1, 1, 1, 1],
                        name="conv"
                    )
                    feature_map = tf.nn.bias_add(
                        z,
                        b_conv,
                        data_format=self.data_format,
                        name="add_b_conv"
                    )
                    feature_map = tf.nn.tanh(feature_map, name="tanh_conv")
                    max_pool_ngram_out = tf.nn.max_pool(
                        feature_map,
                        [1, self.max_seq_len - ngram + 1, 1, 1],
                        strides=self.max_pool_strides,
                        padding="VALID",
                        data_format=self.data_format,
                        name="max_pool"
                    )
                    max_pool_ngram_out = tf.reshape(
                        max_pool_ngram_out,
                        (-1, self.out_channels)
                    )
                    max_pool_out_list.append(max_pool_ngram_out)

            max_pool_out = tf.concat(max_pool_out_list, axis=1)
        # add dropout on max_pool_out
        '''h_dropout = tf.nn.dropout(
            max_pool_out,
            keep_prob=0.5,
            seed=1,
        )'''
        with tf.name_scope("classifier"):
            self.W_ho = tf.get_variable(
                "W_ho",
                (self.num_filters, self.config.n_classes),
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
            pred = tf.matmul(max_pool_out, self.W_ho) + self.b_o

        return pred


class SentenceClassifierSeq2SeqGRUBinary(SentenceClassifier):
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
            (None, self.max_seq_len, self.embedding_size),
            "input"
        )
        self.batch_seq_length_placeholder = tf.placeholder(tf.int32, (None, ),
                                                           "batch_seq_length")
        self.labels_placeholder = tf.placeholder(tf.float32, (None, ), "labels")
        self.dropout_placeholder = tf.placeholder(tf.float32, (), "dropout")


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
        rnn_cell = tf.nn.rnn_cell.GRUCell(
            self.config.rnn_hidden_size,
            activation='relu',
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            bias_initializer=tf.zeros_initializer(),
            name="gru",
            dtype=tf.float32
        )

        rnn_cell_dropout = tf.nn.rnn_cell.DropoutWrapper(
            rnn_cell,
            input_keep_prob=0.9,
            output_keep_prob=1.0,
            state_keep_prob=0.8
        )

        outputs, state = tf.nn.dynamic_rnn(
            cell=rnn_cell_dropout,
            inputs=self.input_placeholder,
            sequence_length=self.batch_seq_length_placeholder,
            dtype=tf.float32
            #initial_state=initial_state
        )

        h_drop = tf.nn.dropout(state, keep_prob=1.0)

        with tf.name_scope("classifier"):
            self.W_ho = tf.get_variable(
                "W_ho",
                (self.config.rnn_hidden_size, 1),
                tf.float32,
                tf.contrib.layers.xavier_initializer(),
                trainable=True
            )
            self.b_o = tf.get_variable(
                "bo",
                (1, 1),
                tf.float32, tf.zeros_initializer(),
                trainable=True
            )
            pred = tf.matmul(h_drop, self.W_ho) + self.b_o
            pred = tf.reshape(pred, (-1, ))
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
        loss = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=self.labels_placeholder,
            logits=pred,
            name="loss"
        )
        loss = tf.reduce_mean(loss)

        return loss

    def add_eval_op(self, pred):
        """Creates f1 evaluator of classifier"""
        f1_score, metric_update_op = tf.contrib.metrics.f1_score(
            self.labels_placeholder,
            tf.nn.sigmoid(self.pred),
            name='f1_score'
        )

        return f1_score, metric_update_op

    def evaluate(self, sess, data_gen, threshold=None):
        """Evaluates model on dev dataset and returns f1_score and predicted
        labels. If threshold is provided, it binarizes predictions, otherwise
        returns softmax output.
        """
        score = None
        pred_labels = np.array([], dtype=np.intp)
        labels = np.array([], dtype=np.intp)
        for inputs, seq_length, batch_labels in data_gen:
            feed_dict = self.create_feed_dict(inputs, seq_length, batch_labels,
                                              self.config.dropout)
            pred = sess.run(self.pred, feed_dict)
            pred = sigmoid(pred)
            if threshold is not None:
                pred = utils.binarize(pred, threshold)

            pred_labels = np.concatenate((pred_labels, pred))
            labels = np.concatenate((labels, batch_labels))

        if threshold is not None:
            score = f1_score(labels, pred_labels)

        return score, pred_labels, labels

    def predict(self, sess, data_gen, threshold=None):
        """Predicts labels on unlabeled test set. If threshold is provided,
        it binarizes predictions, otherwise returns softmax output.
        """
        pred_labels = np.array([], dtype=np.intp)
        for inputs, seq_length in data_gen:
            feed_dict = self.create_feed_dict(inputs, seq_length,
                                              dropout=self.config.dropout)
            pred = sess.run(self.pred, feed_dict)
            pred = sigmoid(pred)
            if threshold is not None:
                pred = utils.binarize(pred, self.threshold)

            pred_labels = np.concatenate((pred_labels, pred))

        return pred_labels.astype(np.int8)
