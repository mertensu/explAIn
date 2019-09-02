import pytest
import numpy as np
from models import FeatureTransformer,RegressionLearner


def test_transformer():
    """
    Test if class FeatureTransformer returns the correct shape.
    :return: None
    """
    inp = np.empty(shape=(64,1))
    transformer = FeatureTransformer()
    assert transformer(inp).numpy().shape == (64,1)


