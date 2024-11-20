import unittest
import pandas as pd
from sklearn.pipeline import Pipeline
from src.model_pipeline import build_pipeline

class TestModelPipeline(unittest.TestCase):
    def test_pipeline_structure(self):
        pipeline = build_pipeline()
        self.assertIsInstance(pipeline, Pipeline)  # Ensure it's a Pipeline
        self.assertTrue('scaler' in pipeline.named_steps)  # Check scaler step
        self.assertTrue('classifier' in pipeline.named_steps)  # Check classifier step

    def test_pipeline_fitting(self):
        df = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [10, 20, 30],
            'quality': [0, 1, 0]
        })
        X = df.drop('quality', axis=1)
        y = df['quality']
        pipeline = build_pipeline()
        pipeline.fit(X, y)
        self.assertTrue(pipeline.named_steps['classifier'].classes_ is not None)  # Ensure model is trained

if __name__ == "__main__":
    unittest.main()

