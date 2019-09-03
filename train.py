import tensorflow as tf
import numpy as np
import os


class Trainer():
    """
    A helper class which provides basic methods to train a model.
    """

    def __init__(self, model, opt, loss_func = None,
                 metric=None, n_epochs=None, n_iters=None,
                 save_path=None):
        """
        :param model: an instance of class RegressionLearner
        :param opt: the optimizer to use
        :param loss_func: the loss function of choice
        :param metric: the metric of choice (has to be tf.keras.metric.?? object)
        :param n_epochs: the number of epochs to train (if dataset_mode)
        :param n_iters: the number of batches to generate (if online_mode)
        :param save_path: should models be saved (default: no (None))
        """
        self.model = model
        self.opt = opt
        self.loss_func = loss_func
        self.metric = metric
        self.n_epochs = n_epochs
        self.n_iters = n_iters
        self.save_path = save_path

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

        if self.save_path is not None:
            self.enable_checkpoint()

        # initialize an instance to calculate weighted means of loss
        self.init_weighted_mean()

        # Run training loop
        for it in range(1, self.n_iters + 1):

            X, y = data_gen(batch_size)
            # forward and backward pass
            self.train_loop(X,y)

            X_val, y_val = data_gen(batch_size)
            # compute validation score
            self.compute_validation_metric(X_val, y_val)

            running_loss = loss_val.numpy() if it < n_smooth else np.mean(losses[-n_smooth:])
            p_bar.set_postfix_str(
                "Iteration: {0}, Training Loss: {2:.3f}, Val Score: {2:.3f}"
                .format(it, running_loss, val_score.numpy()))
            p_bar.update(1)

        return losses

    def init_weighted_mean(self):
        self.train_loss = tf.keras.metrics.Mean(name='train loss')
        self.val_loss = tf.keras.metrics.Mean(name='val loss')

    @tf.function
    def train_loop(self,X,y):
        with tf.GradientTape() as tape:
            y_hat = self.model(X)
            loss_val = self.loss_func(y, y_hat)
        self.backpropagate(tape,loss_val)
        self.train_loss(loss_val)

    def backpropagate(self, tape, loss_val):
        """
        Compute gradients and run update weights.
        :param tape: The tape storing all calculations.
        :param loss_val: the current loss value.
        :return: None
        """
        gradients = tape.gradient(loss_val, self.model.trainable_variables)
        self.opt.apply_gradients(zip(gradients, self.model.trainable_variables))

    @tf.function
    def compute_validation_metric(self, X_val, y_val):
        """
        Compute the metric on a validation batch.
        :param data_gen: The data generator as used for training.
        :param batch_size: The batch size of the validation batch.
        :return: The metric of interest (scalar).
        """
        y_val_hat = self.model(X_val)
        loss_val = self.loss_func(y_val, y_val_hat)
        self.val_loss(loss_val)
        self.metric(y_val, y_val_hat)

    def enable_checkpoint(self):
        """
        Store the last two model checkpoints to easily load model states.
        :return: None
        """
        assert os.path.isdir(self.save_path), 'Directory does not exist'
        checkpoint = tf.train.Checkpoint(optimizer=self.opt, net=self.model)
        manager = tf.train.CheckpointManager(checkpoint, self.save_path, max_to_keep=2)
        checkpoint.restore(manager.latest_checkpoint)
        if manager.latest_checkpoint:
            print("Restored model from {}".format(manager.latest_checkpoint))


