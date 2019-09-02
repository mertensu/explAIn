import tensorflow as tf
import numpy as np


class Trainer():

    def __init__(self, model, opt, loss_func = None, metric=None, n_epochs=None, n_iters=None):
        self.model = model
        self.opt = opt
        self.loss_func = loss_func
        self.metric = metric
        self.n_epochs = n_epochs
        self.n_iters = n_iters

    def run(self):
        pass

    def run_online(self, data_gen, batch_size, p_bar, n_smooth=100):

        losses = []
        # Run training loop
        for it in range(1, self.n_iters + 1):
            with tf.GradientTape() as tape:
                # Generate data and parameters
                X_train, y_train = data_gen(batch_size)
                y_train_hat = self.model(X_train)
                loss_val = self.loss(y_train, y_train_hat)

            losses.append(loss_val.numpy())

            val_score = self.compute_validation_score()

            self.backpropagate(tape, loss_val)

            # Update progress bar
            running_loss = loss_val.numpy() if it < n_smooth else np.mean(losses[-n_smooth:])
            p_bar.set_postfix_str(
                "Iteration: {0}, Training Loss: {2:.3f}, Val Score: {2:.3f}"
                .format(it, running_loss, val_score.numpy()))
            p_bar.update(1)

        return losses

    def backpropagate(self, tape, loss_val):
        gradients = tape.gradient(loss_val, self.model.trainable_variables)
        self.opt.apply_gradients(zip(gradients, self.model.trainable_variables))

    def compute_validation_score(self, data_gen, batch_size):
        X_val, y_val = data_gen(batch_size)
        y_val_hat = self.model(X_val)
        return self.metric(y_val, y_val_hat)

