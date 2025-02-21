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
        """Enhanced feature preparation including engineered features"""
        # Start with base features
        base_features = df[[
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

        # Price Movement Features
        for window in [5, 10, 20]:
            base_features[f'wap_momentum_{window}'] = df.groupby('stock_id')['wap'].pct_change(window).values
            base_features[f'price_volatility_{window}'] = df.groupby('stock_id')['wap'].rolling(window).std().reset_index(0, drop=True).values

        # Order Book Dynamics
        base_features['bid_ask_spread'] = df['ask_price'] - df['bid_price']
        base_features['bid_ask_spread_pct'] = base_features['bid_ask_spread'] / df['wap']
        base_features['imbalance_momentum'] = df.groupby('stock_id')['imbalance_size'].diff().values
        base_features['cumulative_imbalance'] = df.groupby('stock_id')['imbalance_size'].rolling(5).sum().reset_index(0, drop=True).values

        # Volume Analysis
        base_features['total_volume'] = df['bid_size'] + df['ask_size']
        base_features['volume_imbalance'] = df['bid_size'] / df['ask_size']
        for window in [5, 10]:
            base_features[f'volume_momentum_{window}'] = df.groupby('stock_id')['matched_size'].pct_change(window).values
            base_features[f'rolling_volume_mean_{window}'] = df.groupby('stock_id')['matched_size'].rolling(window).mean().reset_index(0, drop=True).values

        # Time Features (if seconds_in_bucket exists in df)
        if 'seconds_in_bucket' in df.columns:
            base_features['seconds_til_close'] = 600 - df['seconds_in_bucket']
            base_features['time_phase'] = df['seconds_in_bucket'] / 600

        # Statistical Features
        for column in ['wap', 'imbalance_size', 'matched_size']:
            means = df.groupby('stock_id')[column].rolling(20).mean().reset_index(0, drop=True)
            stds = df.groupby('stock_id')[column].rolling(20).std().reset_index(0, drop=True)
            base_features[f'{column}_zscore'] = ((df[column] - means) / stds).values

        # Price Relationships
        base_features['far_near_ratio'] = df['far_price'] / df['near_price']
        base_features['mid_price'] = (df['bid_price'] + df['ask_price']) / 2
        base_features['mid_to_wap'] = base_features['mid_price'] / df['wap']

        # Handle missing values and infinities
        base_features = base_features.replace([np.inf, -np.inf], np.nan)
        base_features = base_features.ffill()
        base_features = base_features.bfill()  

        return base_features
            
    def train(self, train_df):
        """Train with offline validation"""
        print("Preparing training data...")
        
        # Debug BEFORE
        print("\nBEFORE Target Calculation:")
        print("Reference price statistics:")
        print(train_df['reference_price'].describe())
        
        # Calculate target
        train_df = calculate_target(train_df)  # Add this line
        
        # Debug AFTER
        print("\nAFTER Target Calculation:")
        print("Target statistics:")
        print(train_df['target'].describe())
        
        # Prepare features
        X = self.prepare_features(train_df)
        y = train_df['target']
        
        # Remove rows with NA
        mask = ~y.isna()
        X = X[mask].copy()
        y = y[mask]
        
        # Split by date for validation
        split_day = train_df['date_id'].max() - 5
        train_dates = train_df[mask]['date_id']
        is_val = train_dates > split_day
        
        X_train = X[~is_val]
        y_train = y[~is_val]
        X_val = X[is_val]
        y_val = y[is_val]
        
        print("Training model...")
        self.model.fit(X_train, y_train)
        
        # Validate offline
        val_pred = self.model.predict(X_val)
        val_score = mean_absolute_error(y_val, val_pred)
        print(f"Validation MAE: {val_score:.4f}")


def calculate_target(df):
        # Calculate stock returns
        df['stock_return'] = df.groupby('stock_id')['wap'].pct_change(-6)  # -6 for 60 seconds ahead
        
        # We need index data for proper calculation
        # df['index_return'] = df.groupby('date_id')['index_wap'].pct_change(-6)
        
        # Convert to basis points
        df['target'] = df['stock_return'] * 10000
        
        return df

def test_with_mock_api():
    # Initialize MockAPI
    api = MockApi()
    api.input_paths = ['./example_test_files/test.csv', 
                      './example_test_files/revealed_targets.csv',
                      './example_test_files/sample_submission.csv']
    api.group_id_column = 'time_id'
    api.export_group_id_column = True
    
    # Load and train model
    print("Loading training data...")
    train_df = pd.read_csv('train.csv')
    
    print("Initializing model...")
    model = OptiverModel()
    model.train(train_df)
    
    print("Starting prediction loop...")
    counter = 0
    predictions = []
    
    for (test, revealed_targets, sample_prediction) in api.iter_test():
        if counter == 0:  # Only for first batch
            print("\nFirst batch debug info:")
            print("Test data features:")
            print(test[['reference_price', 'wap']].describe())
            
            # Make predictions
            X = model.prepare_features(test)
            pred = model.model.predict(X)
            
            print("\nPrediction statistics:")
            print(pd.Series(pred).describe())
            
            print("\nFirst 5 rows comparison:")
            print(pd.DataFrame({
                'prediction': pred[:5],
                'reference_price': test['reference_price'].iloc[:5],
                'wap': test['wap'].iloc[:5]
            }))
        
        # Store revealed targets if they're not empty
        if not revealed_targets.empty and 'target' in revealed_targets.columns:
            # Calculate MAE for previous predictions if possible
            if predictions:
                prev_pred = pd.concat(predictions)
                mae = mean_absolute_error(
                    revealed_targets['target'],
                    prev_pred['target']
                )
                print(f"MAE for previous predictions: {mae:.4f}")
        
        # Prepare features and predict
        X = model.prepare_features(test)
        pred = model.model.predict(X)
        
        # Format predictions
        sample_prediction['target'] = pred
        predictions.append(sample_prediction.copy())
        
        # Submit predictions
        api.predict(sample_prediction)
        counter += 1
    
    print(f"Processed {counter} batches")

if __name__ == "__main__":
    test_with_mock_api()