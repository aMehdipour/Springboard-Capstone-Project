import warnings
import pandas as pd
import numpy as np
import tensorflow as tf
import gradio as gr
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.preprocessing import MinMaxScaler, PowerTransformer

pd.options.mode.chained_assignment = 'warn'

warnings.filterwarnings("ignore")

modelPath = Path('trainedModel/')

model = tf.keras.models.load_model(modelPath)

col_names = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 'T2', 'T24', 'T30', 'T50', 'P2', 'P15', 'P30', 'Nf', 'Nc', 'epr', 'Ps30', 'phi', 'NRf', 'NRc', 'BPR', 'farB', 'htBleed', 'Nf_dmd',
             'PCNfR_dmd', 'W31', 'W32', 's22', 's23']

testData = pd.read_csv("data/test.txt", sep=' ', names=col_names)
testDataNoDrop = pd.read_csv("data/test.txt", sep=' ', names=col_names)
testData = testData.drop(['id', 'cycle', 'setting2', 'setting3', 'T2',
                          'T24', 'epr', 'farB', 'Nf_dmd', 'PCNfR_dmd', 's22', 's23'], axis=1)

gen = MinMaxScaler(feature_range=(0, 1))
pt = PowerTransformer()


def predict(engineId, NRc, T30, P30):
    # W31 is index 15, T30 is index 1, P30 is index 5
    engineIdx = testDataNoDrop.index[testDataNoDrop['id'] == engineId].tolist()
    engineIdx = engineIdx[int(len(engineIdx)/2)]

    testData.loc[engineIdx, 'NRc'] = NRc
    testData.loc[engineIdx, 'T30'] = T30
    testData.loc[engineIdx, 'P30'] = P30

    testDf = gen.fit_transform(testData)
    testDf = pd.DataFrame(testDf)
    testDf = np.nan_to_num(testDf)
    testDf = pt.fit_transform(testDf)
    testDf = np.array(testDf)

    data = testDf[engineIdx]
    data = data.reshape(1, 16)

    pred = int(model.predict(data))

    if pred > 30:
        maintReq = 'No '

    return pred


defaultNrc = int(max(testData['NRc']) - (max(testData['NRc'])-8075)/2)
defaultT = int(max(testData['T30']) - (max(testData['T30'])-1580)/2)
defaultP = int(max(testData['P30']) - (max(testData['P30'])-550)/2)

input = [gr.inputs.Slider(1, 100, step=1, label='Engine ID'),
         gr.inputs.Slider(8075, max(
             testData['NRc']), default=defaultNrc, step=0.1, label='Corrected Engine Core Speed (rpm)'),
         gr.inputs.Slider(1580, max(testData['T30']), default=defaultT,
                          label='Total Temperature at HPC Outlet (\N{DEGREE SIGN}R)'),
         gr.inputs.Slider(550, max(testData['P30']), default=defaultP, label='Total Pressure at HPC Outlet (psi)')]

output = [gr.outputs.Textbox(type='number', label="Remaining Engine Cycles")]
description = "This interface allows you to vary certain sensor readings and\
    uses those to call a neural network trained to predict the amount of\
    remaining life for the specified turbofan engine!\
        Choose your desired engine ID and vary each sensor's\
            readings to see how it affects the engine's remaining life.\
                See the readme file for more details."
title = "Turbofan Engine Reimaining Life Predictor"
title = "Turbofan Engine Reimaining Life Predictor"

iface = gr.Interface(fn=predict, description=description, title=title, inputs=input,
                     outputs=output, live=True, theme="dark-peach")
iface.launch(debug=False)
