import unittest
from click.testing import CliRunner
from Train import train
from pathlib import Path

wrongFeatureData = '../data/wrongmodtrainingdata.csv'
labelData = '../data/traininglabels.csv'


class TestTrain(unittest.TestCase):

    # Make sure that the training, testing data
    # have the correct number of columns.
    def test_pull_and_parse_data(self):
        runner = CliRunner()
        result = runner.invoke(
            train, input='\n'.join([wrongFeatureData, labelData]))
        print(result.__dict__)
        assert isinstance(result.exception, ValueError)
