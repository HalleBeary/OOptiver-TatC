import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from public_timeseries_testing_util import MockApi

class OptiverModel:
    def __init__(self):
        self.model = XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        
    def prepare_features(self, df):
        """Basic feature preparation"""
        features = df[[
            'imbalance_size',
            'imbalance_buy_sell_flag',
            'matched_size',
            'far_price',
            'near_price',
            'wap',
            'bid_price',
            'bid_size',
            'ask_price',
            'ask_size'
        ]].copy()
        
        return features
        
    def prepare_target(self, df):
        """Create target variable (reference_price 60 seconds ahead)"""
        df['target'] = df.groupby('stock_id')['reference_price'].shift(-6)
        return df
    
    def train(self, train_df):
        """Train the model"""
        print("Preparing training data...")
        df = self.prepare_target(train_df)
        X = self.prepare_features(df)
        y = df['target']
        
        # Remove rows with NA
        mask = ~y.isna()
        X = X[mask]
        y = y[mask]
        
        print("Training model...")
        self.model.fit(X, y)
        print("Training completed!")

def test_with_mock_api():
    # Initialize MockAPI
    mock_api = MockApi()
    mock_api.input_paths = ['optiver-trading-at-the-close/example_test_files/test.csv', 'optiver-trading-at-the-close/example_test_files/sample_submission.csv']
    mock_api.group_id_column = 'stock_id'
    mock_api.export_group_id_column = True
    
    # Load training data and train model
    print("Loading training data...")
    train_df = pd.read_csv('optiver-trading-at-the-close/train.csv', nrows=100000)  # Start with subset for testing
    
    print("Initializing model...")
    model = OptiverModel()
    model.train(train_df)
    
    print("Starting prediction loop...")
    predictions = []

        # Debug prints
    print("Checking if files exist...")
    import os
    for path in mock_api.input_paths:
        print(f"Path {path} exists: {os.path.exists(path)}")

    # Try loading first file directly
    print("\nTrying to read test file...")
    test_df = pd.read_csv(mock_api.input_paths[0])
    print("Columns in test file:", test_df.columns.tolist())
    # Add this to your debug section:
    print("\nChecking both files:")
    test_df = pd.read_csv(mock_api.input_paths[0])
    submission_df = pd.read_csv(mock_api.input_paths[1])
    print("Test file columns:", test_df.columns.tolist())
    print("Submission file columns:", submission_df.columns.tolist())
    # Prediction loop


    for test_data in mock_api.iter_test():
        # Get current data
        current_data = test_data[0]
        
        # Prepare features
        X = model.prepare_features(current_data)
        
        # Make prediction
        pred = model.model.predict(X)
        
        # Create prediction dataframe
        pred_df = pd.DataFrame({
            'row_id': current_data['row_id'],
            'target': pred
        })
        
        # Submit prediction
        mock_api.predict(pred_df)
        predictions.append(pred_df)
    
    print("Testing completed!")
    return predictions

if __name__ == "__main__":
    predictions = test_with_mock_api()
    print("First few predictions:")
    print(predictions[0].head())