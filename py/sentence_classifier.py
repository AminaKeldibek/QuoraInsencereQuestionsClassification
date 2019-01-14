import tensorflow as tf


class ModelConfig():
    """Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation. They can then call self.config.<hyperparameter_name> to
    get the hyperparameter settings.
    """
    n_classes = 2
    batch_size = pow(2, 6)
    n_epochs = 2
    lr = 1e-4
    max_gradient_norm = 1  # try with 5
    learning_rate = 0.001
    max_seq_len = 50
    embedding_size = 300
    rnn_hidden_size = 64


class SentenceClassifier():
    def __init__(self, config):
        self.config = config

    def build(self):
        self.add_placeholders()
        self.pred = self.add_prediction_op()
        self.loss = self.add_loss_op(self.pred)
        self.train_op = self.add_training_op(self.loss)

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
        self.dropout_placeholder = tf.placeholder(tf.float32, (), "dropout")

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
        feed_dict[self.dropout_placeholder] = dropout
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


class SentenceClassifierSeq2Seq(SentenceClassifier):
    def add_prediction_op(self):
        """Adds the core transformation for this model which transforms a batch
        of input data into a batch of predictions.

        Calculates forward pass of RNN on input sequence of length Tx:
        h_t = sigmoid(dot(W_hx, x_t) + dot(W_hh, h_(t-1) + b_t)
        After, calculates models prediction from last cell's activation h_Tx:
        pred = dot(h_Tx, W_ho) + b_o

        Returns:
            pred: A tensor of shape (batch_size, n_classes)
        """
        rnn_cell = tf.nn.rnn_cell.BasicRNNCell(self.config.rnn_hidden_size)
        initial_state = rnn_cell.zero_state(self.config.batch_size,
                                            dtype=tf.float32)

        outputs, state = tf.nn.dynamic_rnn(
            cell=rnn_cell,
            inputs=self.input_placeholder,
            sequence_length=self.batch_seq_length_placeholder,
            initial_state=initial_state
        )
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
        pred = tf.matmul(state, self.W_ho) + self.b_o

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

    def add_training_op(self, loss):
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
        optimizer = optimizer.apply_gradients(zip(clipped_gradients, params))

        return optimizer
