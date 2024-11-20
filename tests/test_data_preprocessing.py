import unittest
import pandas as pd
from src.data_preprocessing import preprocess_data

class TestDataPreprocessing(unittest.TestCase):
    def test_preprocess_data(self):
        df = pd.DataFrame({
            'feature1': [1, 2, None],
            'feature2': ['A', 'B', 'C'],
            'quality': [3, 4, 5]
        })
        X, y = preprocess_data(df)
        self.assertEqual(X.isnull().sum().sum(), 0)  # No missing values
        self.assertEqual(len(X), len(y))  # Features and target match in size

if __name__ == "__main__":
    unittest.main()
