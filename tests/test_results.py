import unittest
from txdetect import results


class ResultsTestCase(unittest.TestCase):
    def test_retrieve_old_experiments(self):
        df = results.get_dataframe()
        df = df[(df["version"] == 2.3)
                & (df["rca"] == 0)
                & (df["date"] <= '2018-08-21-11-03')
                & (df["maxangle"] == 0)
                & (df["drop"] == 0)
                & (df["net"] == 4)]
        self.assertEqual(len(df), 10)

    def test_get_crossvalidation_results(self):
        self.assertGreaterEqual(len(results.get_crossvalidation_results()), 7)

    def test_saved_models_location(self):
        df = results.get_dataframe()
        self.assertGreater(len(df), 0)


