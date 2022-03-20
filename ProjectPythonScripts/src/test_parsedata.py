import unittest
from click.testing import CliRunner
from ParseData import pullAndParseData
from pathlib import Path

wrongTrainData = '../data/wrongtraindata.csv'
rightTrainData = '../data/train.txt'
testData = '../data/test.txt'
rulData = '../data/RUL.txt'


class TestParseData(unittest.TestCase):

    # Make sure that the training, testing data
    # have the correct number of columns.
    def test_pull_and_parse_data(self):
        runner = CliRunner()
        result = runner.invoke(
            pullAndParseData, input='\n'.join([wrongTrainData, testData, rulData]))
        print(result.__dict__)
        assert isinstance(result.exception, ValueError)
