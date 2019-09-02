import tensorflow as tf
import numpy as np


class Trainer():
    """
    A helper class which provides basic methods to train a model.
    """

    def __init__(self, model, opt, loss_func = None, metric=None, n_epochs=None, n_iters=None):
        """
        :param model: an instance of class RegressionLearner
        :param opt: the optimizer to use
        :param loss_func: the loss function of choice
        :param metric: the metric of choice
        :param n_epochs: the number of epochs to train (if dataset_mode)
        :param n_iters: the number of batches to generate (if online_mode)
        """
        self.model = model
        self.opt = opt
        self.loss_func = loss_func
        self.metric = metric
        self.n_epochs = n_epochs
        self.n_iters = n_iters

    def run(self):
        pass

    def run_online(self, data_gen, batch_size, p_bar, n_smooth=100):
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

        losses = []
        # Run training loop
        for it in range(1, self.n_iters + 1):
            with tf.GradientTape() as tape:
                # Generate data and parameters
                X_train, y_train = data_gen(batch_size)
                y_train_hat = self.model(X_train)
                loss_val = self.loss(y_train, y_train_hat)

            losses.append(loss_val.numpy())
            val_score = self.compute_validation_score_online()
            self.backpropagate(tape, loss_val)

            running_loss = loss_val.numpy() if it < n_smooth else np.mean(losses[-n_smooth:])
            p_bar.set_postfix_str(
                "Iteration: {0}, Training Loss: {2:.3f}, Val Score: {2:.3f}"
                .format(it, running_loss, val_score.numpy()))
            p_bar.update(1)

        return losses

    def backpropagate(self, tape, loss_val):
        """
        Compute gradients and run update weights.
        :param tape: The tape storing all calculations.
        :param loss_val: the current loss value.
        :return: None
        """
        gradients = tape.gradient(loss_val, self.model.trainable_variables)
        self.opt.apply_gradients(zip(gradients, self.model.trainable_variables))

    def compute_validation_score_online(self, data_gen, batch_size):
        """
        Compute the metric on a validation batch.
        :param data_gen: The data generator as used for training.
        :param batch_size: The batch size of the validation batch.
        :return: The metric of interest (scalar).
        """
        X_val, y_val = data_gen(batch_size)
        y_val_hat = self.model(X_val)
        return self.metric(y_val, y_val_hat)

