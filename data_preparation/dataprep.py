"""Generates and saves a classification dataset as a CSV file."""

import json
import sys
from pathlib import Path

# Add parent directory to path to import logging_config
sys.path.insert(0, str(Path(__file__).parent.parent))
print(f"Added {Path(__file__).parent.parent} to sys.path for module imports.")
print("\nPython sys.path:")
for i, path in enumerate(sys.path, 1):
    print(f"  {i}. {path}")

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from logging_config import get_logger

logger = get_logger("data_preparation", "data_preparation.log", True, "w")

class ClassificationDataHandler:
    """Handles generation and export of synthetic classification datasets."""

    def __init__(self, n_samples=1000, n_features=10, n_classes=2, random_state=42, test_size=0.2):
        """
        Initialize the handler with dataset configuration.

        Args:
            n_samples (int): Number of samples to generate.
            n_features (int): Number of features per sample.
            n_classes (int): Number of target classes.
            random_state (int): Seed for reproducibility.
            test_size (float): Proportion of the dataset to use as test set (0.0 to 1.0).
        """
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_classes = n_classes
        self.random_state = random_state
        self.test_size = test_size
        self.df = None

    def generate_classification_data(self) -> pd.DataFrame:
        """
        Generate a synthetic classification dataset.

        Returns:
            pd.DataFrame: DataFrame with feature columns and a 'target' column.

        Raises:
            Exception: If parameters are invalid or generation fails.
        """
        try:
            X, y = make_classification(
                n_samples=self.n_samples,
                n_features=self.n_features,
                n_classes=self.n_classes,
                random_state=self.random_state,
                n_informative=self.n_features // 2,
                n_redundant=self.n_features // 4,
                n_repeated=self.n_features // 6,
            )
            logger.info(f"Generated classification data with {self.n_samples} samples.")
            logger.info(f"Number of Feature: {self.n_features}")
            logger.info(f"Number of informative features: {self.n_features // 2}")
            logger.info(f"Number of redundant features: {self.n_features // 4}")
            logger.info(f"Number of repeated features: {self.n_features // 6}")
            logger.info(f"Features: {self.n_features}, Classes: {self.n_classes}.")
            feature_names = [f"feature_{i + 1}" for i in range(self.n_features)]
            self.df = pd.DataFrame(X, columns=feature_names)
            # Convert numeric columns to Binned categories before adding target
            self.df = self.bin_numeric_columns(self.df, n_bins=5)
            logger.info("Numeric features binned into categories successfully.")
            self.df["target"] = y
            logger.info("Classification data generated successfully.")
            return self.df
        except ValueError as e:
            error_message = f"Invalid parameter for data generation: {e}"
            logger.error(error_message)
            raise Exception(error_message) from e
        except Exception as e:
            error_message = f"Unexpected error during data generation: {e}"
            logger.error(error_message)
            raise Exception(error_message) from e

    def split_data(self):
        """
        Split the dataset into train and test sets.

        Returns:
            tuple: (X_train, X_test, y_train, y_test) as DataFrames/Series.

        Raises:
            Exception: If no data exists or split parameters are invalid.
        """
        try:
            if self.df is None:
                raise ValueError("No data to split. Run generate_classification_data() first.")
            X = self.df.drop(columns=["target"])
            y = self.df["target"]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state
            )
            logger.info("Data split complete:")
            logger.info(f"  Train samples: {len(X_train)}")
            logger.info(f"  Test samples: {len(X_test)}")
            logger.info(f"  Test size ratio: {self.test_size}")
            logger.info("X_train:")
            logger.info(f"  Shape: {X_train.shape}")
            logger.info(f"  Type: {type(X_train)}")
            logger.info("y_train:")
            logger.info(f"  Shape: {y_train.shape}")
            logger.info(f"  Type: {type(y_train)}")
            logger.info("X_test:")
            logger.info(f"  Shape: {X_test.shape}")
            logger.info(f"  Type: {type(X_test)}")
            logger.info("y_test:")
            logger.info(f"  Shape: {y_test.shape}")
            logger.info(f"  Type: {type(y_test)}")
            return X_train, X_test, y_train, y_test
        except ValueError as e:
            error_message = f"Data split error: {e}"
            logger.error(error_message)
            raise Exception(error_message) from e
        except Exception as e:
            error_message = f"Unexpected error during data split: {e}"
            logger.error(error_message)
            raise Exception(error_message) from e

    def save_data_as_csv(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        output_dir: str = "data",
    ) -> None:
        """
        Save train/test splits as separate CSV files.

        Args:
            X_train (pd.DataFrame): Training features.
            X_test (pd.DataFrame): Test features.
            y_train (pd.Series): Training labels.
            y_test (pd.Series): Test labels.
            output_dir (str): Directory to save the CSV files.

        Raises:
            Exception: If file write fails or an unexpected error occurs.
        """
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            files = {
                "X_train.csv": X_train,
                "X_test.csv": X_test,
                "y_train.csv": y_train,
                "y_test.csv": y_test,
            }

            for filename, data in files.items():
                path = output_path / filename
                data.to_csv(path, index=False)
                logger.info(f"Saved {filename} to {path}")

        except OSError as e:
            error_message = f"Failed to save files to '{output_dir}': {e}"
            logger.error(error_message)
            raise Exception(error_message) from e
        except Exception as e:
            error_message = f"Unexpected error while saving data: {e}"
            logger.error(error_message)
            raise Exception(error_message) from e

    def save_data_schema(
        self,
        X_train: pd.DataFrame,
        output_dir: str = "data",
    ) -> None:
        """
        Extract and save the data schema as a JSON file.

        Args:
            X_train (pd.DataFrame): Training features DataFrame.
            output_dir (str): Directory to save the schema JSON file.

        Raises:
            Exception: If file write fails or an unexpected error occurs.
        """
        try:
            schema = get_data_schema(X_train)
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            schema_path = output_path / "data_schema.json"
            with open(schema_path, 'w', encoding='utf-8') as f:
                json.dump(schema, f, indent=2)

            logger.info(f"Data schema saved to {schema_path}")
        except OSError as e:
            error_message = f"Failed to save data schema to '{output_dir}': {e}"
            logger.error(error_message)
            raise Exception(error_message) from e
        except Exception as e:
            error_message = f"Unexpected error while saving data schema: {e}"
            logger.error(error_message)
            raise Exception(error_message) from e

    def bin_numeric_columns(self, df: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
        """
        Apply quantile-based binning (qcut) to all numeric columns.

        Args:
            df (pd.DataFrame): Input DataFrame.
            n_bins (int): Number of bins for qcut (default: 10).

        Returns:
            pd.DataFrame: DataFrame with numeric columns binned into categories.

        Raises:
            Exception: If binning operation fails.
        """
        try:
            df_binned = df.copy()
            numeric_cols = df_binned.select_dtypes(include=[np.number]).columns.tolist()
            
            # Create clean labels with zero-padded bin number (e.g., bin_01)
            for col in numeric_cols:
                labels = [f"bin_{i:02d}" for i in range(1, n_bins + 1)]
                try:
                    df_binned[col] = pd.qcut(df[col], q=n_bins, labels=labels, duplicates='drop')
                    # Set Data Type as Object
                    df_binned[col] = df_binned[col].astype(object)
                    logger.info(f"Binned column '{col}' into {n_bins} bins with clean labels")
                except Exception as col_error:
                    logger.warning(f"Could not bin column '{col}': {col_error}. Skipping.")
            
            logger.info(f"Binning completed for {len(numeric_cols)} numeric columns")
            return df_binned
        except Exception as e:
            error_message = f"Error during binning operation: {e}"
            logger.error(error_message)
            raise Exception(error_message) from e


def get_data_schema(df: pd.DataFrame) -> dict[str, str]:
    """
    Extract the data schema from a DataFrame.
    
    Maps column names to their original dtypes.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        dict[str, str]: Dictionary mapping column names to their original data types
                        (e.g., 'int64', 'float64', 'object').
    """
    schema = {}
    for col in df.columns:
        dtype = df[col].dtype
        schema[col] = str(dtype)
    
    logger.info("Data schema extracted:")
    for i, (col, dtype) in enumerate(schema.items(), 1):
        logger.info(f"  {i}. {col}: {dtype}")
    
    return schema


if __name__ == "__main__":
    handler = ClassificationDataHandler(
        n_samples=1000,
        n_features=5,
        n_classes=2,
        random_state=42,
        test_size=0.2,
    )

    handler.generate_classification_data()
    X_train, X_test, y_train, y_test = handler.split_data()
    handler.save_data_as_csv(X_train, X_test, y_train, y_test)
    handler.save_data_schema(X_train)

