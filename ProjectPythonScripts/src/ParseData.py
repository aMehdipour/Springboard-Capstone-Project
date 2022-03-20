import numpy as np
import pandas as pd
import logging
import click

from sklearn.preprocessing import MinMaxScaler, PowerTransformer
from utilities import set_logger


@click.command()
@click.option('--trainData', 'trainData', prompt='Enter the full path of the training file: ',
              help='Point to the training data file. Include the filename.txt in the command')
@click.option('--testData', 'testData', prompt='Enter the full path of the test file: ',
              help='Point to the test data file. Include the filename.txt in the command')
@click.option('--rulData', 'rulData', prompt='Enter the full path of the Remaining Useful Life data',
              help='Point to the RUL data file. Include the filename.txt in the command')
def pullAndParseData(trainData, testData, rulData):
    """
    Purpose:
        Function that will read in the raw data required for training, process it
        to comply with the necessary inputs for the model, and then save them
        as .csv files to be used by the model.
    Arguments: 
        trainData, testData, rulData [str]: Pointers to the training, testing, and RUL data.
    Returns:
        None   
    """

    # Configure logging
    logger = set_logger("../logs/ParseData.log")

    # Define the column names for the data
    col_names = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 'T2', 'T24', 'T30', 'T50', 'P2', 'P15', 'P30', 'Nf', 'Nc', 'epr', 'Ps30', 'phi', 'NRf', 'NRc', 'BPR', 'farB', 'htBleed', 'Nf_dmd',
                 'PCNfR_dmd', 'W31', 'W32', 's22', 's23']

    logger.info("-------------------Pulling in data-------------------")
    train = pd.read_csv(f"{trainData}", sep=' ', names=col_names)
    test = pd.read_csv(f"{testData}", sep=' ', names=col_names)
    RUL = pd.read_csv(f"{rulData}", sep=' ', names=['remaining_cycles', 'Nan'])

    if train.shape[1] != len(col_names):
        raise ValueError(
            "Your training data does not contain the correct number of columns")

    if test.shape[1] != len(col_names):
        raise ValueError(
            "Your training data does not contain the correct number of columns")

    # Drop data that contains NaNs
    train.drop(columns=['s22', 's23'], axis=1, inplace=True)
    test.drop(columns=['s22', 's23'], axis=1, inplace=True)
    RUL.drop(columns=['Nan'], axis=1, inplace=True)

    # Process data
    logger.info(
        "-------------------Processing the data-------------------")

    RUL['id'] = RUL.index + 1
    train['remaining_cycles'] = train.groupby(
        ['id'])['cycle'].transform(max)-train['cycle']
    maxCycles = pd.DataFrame(test.groupby('id')['cycle'].max()).reset_index()
    maxCycles.columns = ['id', 'max_tested']
    maxCycles['max_cycles'] = RUL['remaining_cycles'] + maxCycles['max_tested']
    maxCycles.drop(['max_tested'], axis=1, inplace=True)

    test = test.merge(maxCycles, on=['id'], how='left')
    test['remaining_cycles'] = test['max_cycles'] - test['cycle']
    test.drop(['max_cycles'], axis=1, inplace=True)

    X = train.drop(['id', 'cycle', 'setting2', 'setting3', 'T2', 'T24', 'epr', 'farB', 'Nf_dmd', 'PCNfR_dmd',
                    'remaining_cycles'], axis=1)
    X_test = test.drop(['id', 'cycle', 'setting2', 'setting3', 'T2', 'T24', 'epr', 'farB', 'Nf_dmd', 'PCNfR_dmd',
                        'remaining_cycles'], axis=1)

    X_test_non_normal = X_test

    logger.info(
        "-------------------Processing features/labels-------------------")
    gen = MinMaxScaler(feature_range=(0, 1))
    X = gen.fit_transform(X)
    X = pd.DataFrame(X)
    X = np.nan_to_num(X)

    pt = PowerTransformer()
    X = pt.fit_transform(X)
    X = np.array(X)

    gen2 = MinMaxScaler(feature_range=(0, 1))
    X_test = gen.fit_transform(X_test)
    X_test = pd.DataFrame(X_test)
    X_test = np.nan_to_num(X_test)
    pt = PowerTransformer()
    X_test = pt.fit_transform(X_test)
    X_test = np.array(X_test)

    y_test = test.remaining_cycles

    y = train.remaining_cycles

    X = pd.DataFrame(X)
    X_test = pd.DataFrame(X_test)
    y = pd.DataFrame(y)
    y_test = pd.DataFrame(y_test)
    # Export data to respective .csv files
    logger.info(
        "-------------------Exporting to .csv files-------------------")

    X.to_csv('../data/trainingdata.csv')
    X_test.to_csv('../data/testdata.csv')
    y.to_csv('../data/traininglabels.csv')
    y_test.to_csv('../data/testinglabels.csv')
    X_test_non_normal.to_csv('../data/unnormalizedTestData.csv')

    logger.info(
        "-------------------Files have been exported-------------------")
    logger.info(
        "-------------------Check the data folder-------------------")


if __name__ == '__main__':

    pullAndParseData()
