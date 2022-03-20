import logging
import click
import pandas as pd
import numpy as np
import tensorflow as tf

from pathlib import Path
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from sklearn.model_selection import KFold, GridSearchCV, RandomizedSearchCV, train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, r2_score, roc_auc_score, precision_score, recall_score, f1_score, mean_squared_error
from utilities import set_logger


@click.command()
@click.option('--featureData', 'featureData', prompt='Enter the full path of the modified feature csv: ',
              help='Point to the processed training data files.')
@click.option('--labelData', 'labelData', prompt='Enter the full path of the modified label csv: ',
              help='Point to the processed training data files.')
def train(featureData, labelData):
    """
    Purpose:
        Train and return a model based on input training data
    Arguments:
        trainData [str]: Path to processed training dataset
    Returns:
        None
    """

    # Configure logging
    logger = set_logger("../logs/train.log")

    # Load in data
    logger.info(
        f"-------------------Loading the processed data-------------------")

    X = pd.read_csv(Path(featureData))
    X = X.drop(columns='Unnamed: 0')
    X = np.array(X)
    y = pd.read_csv(Path(labelData))
    y = y.drop(columns='Unnamed: 0')

    # Check to make sure X and Y have the same number of data points
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            "Your features and labels do not contain the same amount of values")

    logger.info(
        f"-------------------Creating and training AutoEncoder-------------------")

    # Create and train AutoEncoder
    logger.info(
        f"-------------------Creating and training AutoEncoder-------------------")

    class AutoEncoder(Model):
        def __init__(self, latent_dim, input_dim):
            super(AutoEncoder, self).__init__()
            self.latent_dim = latent_dim
            self.encoder = tf.keras.Sequential([
                # layers.Dense(16*latent_dim, activation = "relu"),
                # layers.Dense(8*latent_dim, activation = "relu"),
                # layers.Dense(4*latent_dim, activation = "relu"),
                layers.Dense(2*latent_dim, activation="relu"),
                layers.Dense(latent_dim, activation="relu"),
            ])
            self.decoder = tf.keras.Sequential([
                layers.Dense(2*latent_dim, activation="relu"),
                # layers.Dense(4*latent_dim, activation = "relu"),
                # layers.Dense(8*latent_dim, activation = "relu"),
                # layers.Dense(16*latent_dim, activation = "relu"),
                layers.Dense(input_dim, activation="relu"),
            ])

        def call(self, x):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return decoded

    logger.info(
        f"-------------------Training AutoEncoder-------------------")

    latent_dim = 11
    tf.keras.backend.clear_session()
    callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=0, patience=10)
    autoencoder = AutoEncoder(latent_dim, X.shape[1])
    autoencoder.compile(optimizer="adam", loss="mean_squared_error")
    autoencoder.build(X.shape)

    autoencoder.fit(X, X, epochs=1000, shuffle=True, validation_data=(
        X, X), verbose=True, callbacks=callback)
    autoencoder.encoder.trainable = False

    logger.info(
        f"-------------------AutoEncoder trained!-------------------")
    # logger.info(
    #     f"-------------------Loss: {history.history['loss'][-1]}-------------------")
    # logger.info(
    #     f"-------------------Validation Loss: {history.history['val_loss'][-1]}-------------------")

    # Create fully connected network with AutoEncoder input
    logger.info(
        f"-------------------Creating and training fully connected net-------------------")

    dnn = make_fully_connected_model(autoencoder, 'mean_squared_error', 'relu')
    callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=0, patience=10)
    dnn.fit(X, y, validation_split=0.1886, verbose=1,
            epochs=200, callbacks=callback)

    logger.info(
        f"-------------------Fully connected net trained!-------------------")

    # Save model
    logger.info(
        f"-------------------Saving model-------------------")

    dnn.save('../model/trainedModel')

    logger.info(
        f"-------------------Model saved!-------------------")


def make_fully_connected_model(autoencoder, loss_type, activation_type):
    """
    Purpose:
        Create and compile a fully connected network with an AutoEncoder input
    Arguments:
        autoencoder [tensorflow model]: Trained, frozen AutoEncoder class
        loss_type [str]: Type of loss function to be used during training
        activation_type [str]: Activation function to be applied to layers
    Returns:
        model: Compiled tensorflow model
    """
    autoencoder.encoder.trainable = False
    model = keras.Sequential([autoencoder.encoder,
                              layers.Dense(
                                  512, kernel_initializer="glorot_normal", activation=activation_type),
                              layers.BatchNormalization(axis=1),
                              layers.Dense(
                                  256, kernel_initializer="glorot_normal", activation=activation_type),
                              layers.BatchNormalization(axis=1),
                              layers.Dense(
                                  128, kernel_initializer="glorot_normal", activation=activation_type),
                              layers.BatchNormalization(axis=1),
                              layers.Dense(
                                  64, kernel_initializer="glorot_normal", activation=activation_type),
                              layers.Dropout(0.5),
                              layers.BatchNormalization(axis=1),
                              layers.Dense(
                                  16, kernel_initializer="glorot_normal", activation=activation_type),
                              layers.Dense(1, kernel_initializer="glorot_normal", activation="linear")])
    model.compile(loss=loss_type, optimizer=keras.optimizers.Adam(
        learning_rate=0.001))
    return model


if __name__ == '__main__':
    train()
