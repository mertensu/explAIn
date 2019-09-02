import tensorflow as tf
import numpy as np
from models import RegressionLearner
import pytest

if __name__ == '__main__':
    tf.enable_eager_execution()

    pytest.main()

    dataset = np.empty(shape=(64, 5))
    reglearner = RegressionLearner(5)

    predictions = reglearner(dataset)
    print(predictions.shape)


