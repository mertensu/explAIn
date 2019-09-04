import tensorflow as tf
import itertools


class FeatureTransformer(tf.keras.Model):
    """
    Non-linear transformation of a feature via simple NN.
    """
    def __init__(self, n_layers=2, n_units=3):
        """
        Initialise dense net as the non-linear transformation.
        :param n_layers: number of layers of the net.
        :param n_units: number of hidden units in each layer.
        """
        super(FeatureTransformer, self).__init__()
        self.model = tf.keras.Sequential(
            [tf.keras.layers.Dense(n_units, activation='relu')
             for i in range(n_layers)] +
            [tf.keras.layers.Dense(1)] +
            [tf.keras.layers.BatchNormalization()]
        )

    def call(self, x, training=True):
        """
        Run input through dense layers.
        :param x: input of shape (batch_size, 1)
        :param training: how to normalize input
        :return: transformed input of shape (batch_size, 1)
        """
        return self.model(x,training=training)


class RegressionLearner(tf.keras.Model):
    """
    Implementation of the 'generalized additive machine'
    """
    def __init__(self, n_features, build_interactions=True):
        """
        :param n_features: number of features in the data.
        :param build_interactions: should two-way interaction be included; default: True
        """
        super(RegressionLearner, self).__init__()
        self.nfeatures = n_features
        self.nints = 0
        if build_interactions:
            self.nints = len(self.determine_ninteractions())
        self.transformers = [FeatureTransformer() for i in range(n_features + self.nints)]
        self.regression = tf.keras.layers.Dense(1)

    def determine_ninteractions(self):
        """
        Generate a list of all possible pairwise combinations of column indices.
        :return: a list of tuples with indices building the interaction
        """
        return list(itertools.combinations(range(self.nfeatures), 2))

    def construct_interactions(self, x):
        """
        Generate a list of tensors with concatenated features.
        :param x: input of shape (batch_size, n_features)
        :return: list of tensors each of shape (batch_size, 2).
        """
        out = []
        combs = self.determine_ninteractions()
        for col_idx1, col_idx2 in combs:
            out.append(tf.concat([
                tf.expand_dims(x[:, col_idx1], 1), tf.expand_dims(x[:, col_idx2], 1)
            ], axis=-1))
        return out

    def call(self, x):
        """
        Transform each individual feature and each interaction term. Afterwards,
        run a simple multiple regression using the transformed features as input.
        :param x: input of shape (batch_size, n_features)
        :return: predictions of shape (batch_size, 1)
        """
        # transform single features
        transformed_features = tf.stack([self.transformers[i](x[:, i][:, tf.newaxis])
                                         for i in range(self.nfeatures)], axis=1)[:, :, 0]

        if self.nints > 0:
            # build list of tensors with concatenated features
            x_int = self.construct_interactions(x)

            # transform interactions terms
            transformed_ints = tf.stack([self.transformers[i](x_int[i - self.nfeatures])
                                     for i in range(self.nfeatures, self.nfeatures + self.nints)], axis=1)[:, :, 0]

            # run simple dense layer (multiple regression)
            return self.regression(tf.concat([transformed_features, transformed_ints], axis=-1))

        return self.regression(transformed_features)
