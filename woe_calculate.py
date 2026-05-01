import numpy as np
import json
import pandas as pd
from logging_config import get_logger

logger = get_logger("woe_calculate", "woe_calculate.log", True, "w")

class WoeCalculate:
    def __init__(self):
        pass

    def calculate_woe(self, X, y, feature_col, value_mapping=None):
        """
        Calculate WOE and IV for a given feature.
        
        Args:
            X: Features DataFrame
            y: Target Series
            feature_col: Feature column name
            value_mapping: Dict mapping values to class labels. 
                          Default: {0: 'good', 1: 'bad'}
                          Example: {0: 'good', 1: 'bad'} means y=0 is good, y=1 is bad
        """
        if value_mapping is None:
            value_mapping = {0: 'good', 1: 'bad'}
        
        # Identify good and bad values from mapping
        good_value = [k for k, v in value_mapping.items() if v == 'good'][0]
        bad_value = [k for k, v in value_mapping.items() if v == 'bad'][0]
        
        logger.info(f"Calculating WOE for feature: {feature_col}")
        logger.info(f"Value mapping: Good={good_value}, Bad={bad_value}")
        
        # Calculate total good and bad counts
        total_good = (y == good_value).sum()
        total_bad = (y == bad_value).sum()
        logger.info(f"Total good: {total_good}, Total bad: {total_bad}")
        
        # Group by the feature column and calculate good and bad counts
        grouped = X.join(y).groupby(feature_col)
        grouped_agg = grouped[y.name].agg(
            goods=lambda x: (x == good_value).sum(),
            bads=lambda x: (x == bad_value).sum(),
            count='count'
        )
        grouped_df = grouped_agg.reset_index()
        grouped_df = grouped_df.rename(columns={feature_col: 'Bins'})
        
        # Calculate distributions
        grouped_df['TotalDistr'] = (grouped_df['count'] / grouped_df['count'].sum() * 100).round(2)
        grouped_df['DistrGood'] = (grouped_df['goods'] / total_good * 100).round(2)
        grouped_df['DistrBad'] = (grouped_df['bads'] / total_bad * 100).round(2)

        # Raise Error if DistrBad is zero to avoid division by zero in Odds Ratio calculation
        if (grouped_df['DistrBad'] == 0).any():
            # Log Number of Goods and Bads for each bin to help identify which bin(s) have zero bads
            logger.error(f"Distribution of bads (DistrBad) contains zero values for feature {feature_col}. This will lead to division by zero in Odds Ratio calculation.")
            logger.error(f"Number of Goods and Bads for each bin:\n{grouped_df[['Bins', 'goods', 'bads']]}")
            raise ValueError("Distribution of bads (DistrBad) contains zero values, which will lead to division by zero in Odds Ratio calculation.\
                              Please check the data and ensure that there are no bins with zero bads.")
        
        # Raise Error if DistrGood is zero to avoid division by zero in Odds Ratio calculation
        if (grouped_df['DistrGood'] == 0).any():
            # Log Number of Goods and Bads for each bin to help identify which bin(s) have zero goods
            logger.error(f"Distribution of goods (DistrGood) contains zero values for feature {feature_col}. This will lead to division by zero in Odds Ratio calculation.")
            logger.error(f"Number of Goods and Bads for each bin:\n{grouped_df[['Bins', 'goods', 'bads']]}")
            raise ValueError("Distribution of goods (DistrGood) contains zero values, which will lead to division by zero in Odds Ratio calculation.\
                              Please check the data and ensure that there are no bins with zero goods.") 

        # Calculate Odds Ratio (Distribution ratio: Good% / Bad%)
        grouped_df['OddsRatio'] = (grouped_df['DistrGood'] / grouped_df['DistrBad']).round(4)
        
        # Calculate WOE: ln(OddsRatio) = ln(% Goods / % Bads)
        grouped_df['WOE'] = np.log(grouped_df['OddsRatio']).round(4)
        
        # Calculate IV: (% Goods - % Bads) * WOE
        grouped_df['IV'] = ((grouped_df['DistrGood'] - grouped_df['DistrBad']) / 100 * grouped_df['WOE']).round(4)
        
        # Select and reorder columns
        result = grouped_df[['Bins', 'count', 'TotalDistr', 'goods', 'DistrGood', 'bads', 'DistrBad', 'OddsRatio', 'WOE', 'IV']]
        result = result.rename(columns={
            'count': 'Count',
            'goods': 'Goods',
            'bads': 'Bads',
            'OddsRatio': 'Odds-Ratio'
        })
        
        logger.info(f"WOE Analysis for feature \n{feature_col}:\n\n{result}\n")

        is_woe_monotonic = result['WOE'].is_monotonic_increasing or result['WOE'].is_monotonic_decreasing
        logger.info(f"Is WOE monotonic for feature {feature_col}:\n\t{'Yes' if is_woe_monotonic else 'No'}")

        logger.info(f"Total IV for feature {feature_col}:\n\t{result['IV'].sum():.4f}")
        comment_for_iv = None
        if result['IV'].sum() < 0.02:
            comment_for_iv = "Not Predictive"
        elif 0.02 <= result['IV'].sum() < 0.1:
            comment_for_iv = "Weak Predictive Power"
        elif 0.1 <= result['IV'].sum() < 0.3:
            comment_for_iv = "Medium Predictive Power"
        elif 0.3 <= result['IV'].sum() < 0.5:
            comment_for_iv = "Strong Predictive Power"
        else:
            comment_for_iv = "Suspiciously High IV (Check for Data Leakage)"
        logger.info(f"Comment for IV:\n\t{comment_for_iv}")

        return result
    
    def assign_woe_to_bins(self, woe_result, feature_name, X):
        """
        Assign WOE values to the original feature column based on bins.
        
        Args:
            woe_result: DataFrame returned from calculate_woe() method
            feature_name: Feature column name
            X: Features DataFrame containing the original feature column
            
        Returns:
            pd.Series: Series with WOE values assigned to each row
        """
        logger.info(f"Assigning WOE values to feature: {feature_name}")
        
        # Create a copy of X to avoid modifying the original
        X_copy = X.copy()
        
        # Create a mapping dictionary from Bins to WOE values
        woe_mapping = dict(zip(woe_result['Bins'], woe_result['WOE']))
        
        logger.info(f"WOE Mapping for {feature_name}:\n{woe_mapping}")
        
        # Map the feature values to their corresponding WOE values
        woe_series = X_copy[feature_name].map(woe_mapping)
        
        # Check for any missing values that couldn't be mapped
        if woe_series.isna().any():
            unmapped_values = X_copy[feature_name][woe_series.isna()].unique()
            logger.warning(f"Found {woe_series.isna().sum()} unmapped values for feature {feature_name}: {unmapped_values}")
        
        logger.info(f"WOE values assigned to feature {feature_name}:\n{woe_series.head()}")
        
        return woe_series
    
if __name__ == "__main__":
    # Read Data Schema as a JSON file
    data_schema_path = "data/data_schema.json"
    # Read Json File
    with open(data_schema_path, 'r') as f:
        data_schema = json.load(f)
    
    # Read Data from data folder with data schema without index column
    X_train = pd.read_csv("data/X_train.csv", dtype=data_schema)
    y_train = pd.read_csv("data/y_train.csv").squeeze()  # Convert to Series

    # Define value mapping: 0 = good, 1 = bad
    value_mapping = {0: 'good', 1: 'bad'}

    # Calculate WOE for each feature in X_train
    woe_calculator = WoeCalculate()

    woe_df = pd.DataFrame()  # Initialize an empty DataFrame to store WOE results for all features
    for feature_name in X_train.columns:
        # Calculate WOE and IV for the feature
        woe_result = woe_calculator.calculate_woe(X_train, y_train, feature_name, value_mapping=value_mapping)
        # Assign WOE values to the original feature column
        woe_series = woe_calculator.assign_woe_to_bins(woe_result, feature_name, X_train)
        # Save WOE transformed feature to a new CSV file
        woe_df[feature_name] = woe_series
    # Save the WOE transformed features to a new CSV file
    woe_df.to_csv("data/X_train_woe.csv", index=False)