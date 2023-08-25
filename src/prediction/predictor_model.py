import os
import warnings
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError

warnings.filterwarnings("ignore")


PREDICTOR_FILE_NAME = "predictor.joblib"


class Classifier:
    """A wrapper class for the Random Forest classifier.

    This class provides a consistent interface that can be used with other
    classifier models.
    """

    model_name = "random_forest_classifier"

    def __init__(
        self,
        n_estimators: Optional[int] = 100,
        min_samples_split: Optional[int] = 2,
        min_samples_leaf: Optional[int] = 1,
        **kwargs,
    ):
        """Construct a new Random Forest Classifier.

        Args:
            n_estimators (int, optional): The number of trees in the forest.
                Defaults to 100.
            min_samples_split (int, optional): The minimum number of samples required
                to split an internal node.
                Defaults to 2.
            min_samples_leaf (int, optional): The minimum number of samples required
                to be at a leaf node.
                Defaults to 1.
        """
        self.n_estimators = int(n_estimators)
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.model = self.build_model()
        self._is_trained = False

    def build_model(self) -> RandomForestClassifier:
        """Build a new Random Forest classifier."""
        model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=0,
        )
        return model

    def fit(self, train_inputs: pd.DataFrame, train_targets: pd.Series) -> None:
        """Fit the Random Forest classifier to the training data.

        Args:
            train_inputs (pandas.DataFrame): The features of the training data.
            train_targets (pandas.Series): The labels of the training data.
        """
        self.model.fit(train_inputs, train_targets)
        self._is_trained = True

    def predict(self, inputs: pd.DataFrame) -> np.ndarray:
        """Predict class labels for the given data.

        Args:
            inputs (pandas.DataFrame): The input data.
        Returns:
            numpy.ndarray: The predicted class labels.
        """
        return self.model.predict(inputs)

    def predict_proba(self, inputs: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities for the given data.

        Args:
            inputs (pandas.DataFrame): The input data.
        Returns:
            numpy.ndarray: The predicted class probabilities.
        """
        return self.model.predict_proba(inputs)

    def evaluate(self, test_inputs: pd.DataFrame, test_targets: pd.Series) -> float:
        """Evaluate the Random Forest classifier and return the accuracy.

        Args:
            test_inputs (pandas.DataFrame): The features of the test data.
            test_targets (pandas.Series): The labels of the test data.
        Returns:
            float: The accuracy of the Random Forest classifier.
        """
        if self.model is not None:
            return self.model.score(test_inputs, test_targets)
        raise NotFittedError("Model is not fitted yet.")

    def save(self, model_dir_path: str) -> None:
        """Save the Random Forest classifier to disk.

        Args:
            model_dir_path (str): Dir path to which to save the model.
        """
        if not self._is_trained:
            raise NotFittedError("Model is not fitted yet.")
        joblib.dump(self, os.path.join(model_dir_path, PREDICTOR_FILE_NAME))

    @classmethod
    def load(cls, model_dir_path: str) -> "Classifier":
        """Load the Random Forest classifier from disk.

        Args:
            model_dir_path (str): Dir path to the saved model.
        Returns:
            Classifier: A new instance of the loaded Random Forest classifier.
        """
        model = joblib.load(os.path.join(model_dir_path, PREDICTOR_FILE_NAME))
        return model

    def __str__(self):
        # sort params alphabetically for unit test to run successfully
        return (
            f"Model name: {self.model_name} ("
            f"min_samples_leaf: {self.min_samples_leaf}, "
            f"min_samples_split: {self.min_samples_split}, "
            f"n_estimators: {self.n_estimators})"
        )


def train_predictor_model(
    train_inputs: pd.DataFrame, train_targets: pd.Series, hyperparameters: dict
) -> Classifier:
    """
    Instantiate and train the predictor model.

    Args:
        train_X (pd.DataFrame): The training data inputs.
        train_y (pd.Series): The training data labels.
        hyperparameters (dict): Hyperparameters for the classifier.

    Returns:
        'Classifier': The classifier model
    """
    classifier = Classifier(**hyperparameters)
    classifier.fit(train_inputs=train_inputs, train_targets=train_targets)
    return classifier


def predict_with_model(
    classifier: Classifier, data: pd.DataFrame, return_probs=False
) -> np.ndarray:
    """
    Predict class probabilities for the given data.

    Args:
        classifier (Classifier): The classifier model.
        data (pd.DataFrame): The input data.
        return_probs (bool): Whether to return class probabilities or labels.
            Defaults to True.

    Returns:
        np.ndarray: The predicted classes or class probabilities.
    """
    if return_probs:
        return classifier.predict_proba(data)
    return classifier.predict(data)


def save_predictor_model(model: Classifier, predictor_dir_path: str) -> None:
    """
    Save the classifier model to disk.

    Args:
        model (Classifier): The classifier model to save.
        predictor_dir_path (str): Dir path to which to save the model.
    """
    if not os.path.exists(predictor_dir_path):
        os.makedirs(predictor_dir_path)
    model.save(predictor_dir_path)


def load_predictor_model(predictor_dir_path: str) -> Classifier:
    """
    Load the classifier model from disk.

    Args:
        predictor_dir_path (str): Dir path where model is saved.

    Returns:
        Classifier: A new instance of the loaded classifier model.
    """
    return Classifier.load(predictor_dir_path)


def evaluate_predictor_model(
    model: Classifier, x_test: pd.DataFrame, y_test: pd.Series
) -> float:
    """
    Evaluate the classifier model and return the accuracy.

    Args:
        model (Classifier): The classifier model.
        x_test (pd.DataFrame): The features of the test data.
        y_test (pd.Series): The labels of the test data.

    Returns:
        float: The accuracy of the classifier model.
    """
    return model.evaluate(x_test, y_test)
