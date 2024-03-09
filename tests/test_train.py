import numpy as np

from src.train import train_model


def test_pipeline():
    """Test the pipeline's behavior"""

    # Create some test data
    X_train = np.random.rand(7,789)
    y_train = np.array([1, 2, 3, 0, 4, 6, 5, 7])
    X_test = np.random.rand(5,789)
    # Fit the pipeline on the training data
    model=train_model(X_train, y_train)

    # Test the pipeline's behavior on the test data
    y_pred = model.predict(X_test)

    # Check that the pipeline's output is of the correct shape
    assert y_pred.shape == (5,)

    # Check that the pipeline's output is not all zeros
    assert np.any(y_pred)

    # Check that the pipeline's output is within a reasonable range
    assert np.all(y_pred >= 0) and np.all(y_pred <= 7)