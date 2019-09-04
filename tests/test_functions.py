import tensorflow as tf
import numpy as np
from models import FeatureTransformer, RegressionLearner


def test_transformer_shape():
    """
    Test if class FeatureTransformer returns the correct shape.
    :return: None
    """
    batch_size = 64
    inp = np.empty(shape=(batch_size, 1))
    transformer = FeatureTransformer()
    assert transformer(inp).numpy().shape == (batch_size, 1)


def test_reglearner_shape():
    """
    Test if class RegressionLearner returns the correct shape.
    :return: None
    """
    batch_size, n_features = 64, 5
    dataset = np.empty(shape=(batch_size, n_features))
    reglearner = RegressionLearner(n_features)
    predictions = reglearner(dataset)
    assert predictions.shape == (batch_size, 1)
