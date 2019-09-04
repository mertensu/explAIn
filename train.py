import tensorflow as tf
from fastprogress import master_bar, progress_bar
import os


class Trainer():
    """
    A helper class which provides basic methods to train a model.
    """

    def __init__(self, model, opt, loss_func = None,
                 metric=None, n_epochs=None, n_iters_per_epoch=None,
                 save_path=None):
        """
        :param model: an instance of class RegressionLearner
        :param opt: the optimizer to use
        :param loss_func: the loss function of choice
        :param metric: the metric of choice (has to be tf.keras.metric.?? object)
        :param n_epochs: the number of epochs to train (if dataset_mode)
        :param n_iters_per_epoch: the number of batches to generate per epoch (only in online-mode)
        :param save_path: should models be saved (default: no (None))
        """
        self.model = model
        self.opt = opt
        self.loss_func = loss_func
        self.metric = metric
        self.n_epochs = n_epochs
        self.n_iters = n_iters_per_epoch
        self.save_path = save_path

        self.checkpoint = None
        self.manger = None

    def run(self):
        pass

    def run_online(self, data_gen, batch_size):
        """
        Train a model using online-mode, i.e. if a function to generate
        batches of data is available.
        :param data_gen: A data generator, the function to yield batches.
        The function has to have one argument (batch_size).
        :param batch_size: batch size
        :param p_bar: a progress bar
        :param n_smooth: the number of losses to consider for computation of running loss.
        :return: a list of training losses
        """

        if self.n_iters is None:
            raise ValueError('If you want to run online-mode, please specify n_iters_per_epoch')

        if self.save_path is not None:
            self.check_for_previous_models()

        # initialize an instance to calculate weighted means of loss
        self.init_aggregators()

        mb = master_bar(range(1, self.n_epochs + 1))
        for epoch in mb:

            # Run training loop
            for it in progress_bar(range(1, self.n_iters + 1), parent=mb):

                X, y = data_gen(batch_size)
                # forward and backward pass
                self.train_loop(X,y)

                X_val, y_val = data_gen(batch_size)
                # compute validation score
                self.compute_validation_metric(X_val, y_val)

            mb.write(f'Epoch {epoch}, '
                     f'Loss: {self.train_loss.result():.3f}, '
                     f'Val metric: {self.metric.result():.3f}')

            # Reset the metrics for the next epoch
            self.reset_aggregators()

    def init_aggregators(self):
        """
        Initialize an instance to compute the weighted mean of the
        training loss and the validation metric.
        :return: None
        """
        self.train_loss = tf.keras.metrics.Mean(name='train loss')
        self.val_loss = tf.keras.metrics.Mean(name='val loss')

    def reset_aggregators(self):
        """
        Reset the weighted mean of the training loss and the validation metric
        to start anew in the next epoch.
        :return: None
        """
        self.train_loss.reset_states()
        self.metric.reset_states()

    @tf.function
    def train_loop(self,X,y):
        """
        Run forward and backward pass for a single batch and update
        aggregator.
        :param X: a tensor of shape (batch_size, n_features)
        :param y: a tensor of shape (batch_size, 1) containing the true labels
        :return: None
        """
        with tf.GradientTape() as tape:
            y_hat = self.model(X)
            loss_val = self.loss_func(y, y_hat)
        self.backpropagate(tape,loss_val)
        self.train_loss(loss_val)

    def backpropagate(self, tape, loss_val):
        """
        Compute gradients and update weights.
        :param tape: The tape storing all calculations.
        :param loss_val: the current loss value.
        :return: None
        """
        gradients = tape.gradient(loss_val, self.model.trainable_variables)
        self.opt.apply_gradients(zip(gradients, self.model.trainable_variables))

    @tf.function
    def compute_validation_metric(self, X_val, y_val):
        """
        Compute the metric on a validation batch and update aggregator.
        :param data_gen: The data generator as used for training.
        :param batch_size: The batch size of the validation batch.
        :return: None
        """
        y_val_hat = self.model(X_val)
        loss_val = self.loss_func(y_val, y_val_hat)
        self.val_loss(loss_val)
        self.metric(y_val, y_val_hat)

    def check_for_previous_models(self):
        """
        Check if architecture was already trained and load model state.
        Store the last two model checkpoints to easily load model states.
        :return: None
        """
        assert os.path.isdir(self.save_path), 'Directory does not exist'
        self.checkpoint = tf.train.Checkpoint(optimizer=self.opt, net=self.model)
        self.manager = tf.train.CheckpointManager(self.checkpoint, self.save_path, max_to_keep=2)
        self.checkpoint.restore(self.manager.latest_checkpoint)
        if self.manager.latest_checkpoint:
            print("Loading previously trained model from {}".format(self.manager.latest_checkpoint))
        else:
            print('Initializing manager to store model states after each epoch.')


