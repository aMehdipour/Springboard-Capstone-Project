import tensorflow as tf
import numpy as np
import pandas as pd
import gradio as gr
import click

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from pathlib import Path


X = pd.read_csv('../data/trainingdata.csv')
X = X.iloc[-1, :]


@click.command()
@click.option('--modelPath')
def deploy(modelPath, x):
    """
    Purpose:
        Import trained model and deploy it on Gradio, taking user inputs for certain features
    Arguments:
        modelPath [str]: The path to the trained model
        feature1  [float]: User input value of feature 1
        feature2  [float]: User input value of feature 2
        feature3  [float]: User input value of feature 3
        x         [pd.DataFrame]: Values of one timestep of data
    """
