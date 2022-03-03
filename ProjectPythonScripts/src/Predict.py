import logging
import click
import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.metrics import confusion_matrix, accuracy_score, r2_score, roc_auc_score, precision_score, recall_score, f1_score, mean_squared_error
from utilities import set_logger, load_model
from pathlib import Path


@click.command()
@click.option('--testData', 'testData', prompt='Enter the full path of the testing data files: ',
              help='Point to the processed testing data files.')
@click.option('--modelPath', 'modelPath', prompt='Enter the full path of the fully trained model: ',
              help='Point to the fully trained model.')
def predict(testData, modelPath):
    """
    Purpose:
        Run the test dataset through the trained model, create metrics
    Arguments:
        testData [str]: The full path of the test dataset
        modelPath [str]: The full path of the trained model
    Returns:
        None
    """
    # Configure logging
    logger = set_logger("../logs/predict.log")

    # Load in the trained model
    logger.info(
        f"-------------------Loading the trained model------------------")

    dnn = load_model(modelPath)

    logger.info(
        f"-------------------Trained model loaded!------------------")

    # Load in the test dataset
    logger.info(
        f"-------------------Loading the pre-processed testing data------------------")

    filePath = Path(testData)
    xPath = filePath / "testdata.csv"
    yPath = filePath / "testinglabels.csv"

    X_test = pd.read_csv(xPath)
    X_test = X_test.drop(columns='Unnamed: 0')
    X_test = np.array(X_test)
    y_test = pd.read_csv(yPath)
    y_test = y_test.drop(columns='Unnamed: 0')

    # Check to make sure X and Y have the same number of data points
    assert X_test.shape[0] == y_test.shape[0]

    # Evaluate test dataset
    logger.info(
        f"-------------------Evaluating test dataset------------------")
    dnn.evaluate(X_test, y_test, verbose=1)

    # Make predictions
    logger.info(
        f"-------------------Evaluating test dataset------------------")

    dnn.evaluate(X_test, y_test, verbose=1)
    pred = dnn.predict(X_test)
    pred = np.reshape(pred, -1)

    logger.info(
        f"-------------------R^2 is {r2_score(y_test, pred)}------------------")
    MSE = np.mean((pred-y_test)**2)
    logger.info(
        f"-------------------MSE is {MSE}------------------")


if __name__ == '__main__':
    predict()
