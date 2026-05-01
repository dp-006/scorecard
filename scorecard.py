import numpy as np
import pandas as pd
import pickle
import os
import json
import statsmodels.api as sm
from logging_config import get_logger

logger = get_logger("scorecard", "scorecard.log", True, "w")


class Scorecard:
    def __init__(self, pdo=20, base_score=600, base_odds=50):
        """
        Credit Risk Scorecard class
        Score = Offset + Factor * ln(odds)
        
        Args:
            base_score: Base score
            base_odds: Base odds
            pdo: Points to Double the Odds
        """
        logger.info("="*50)
        logger.info("Scorecard initialized")
        self.pdo = pdo
        self.base_score = base_score
        self.base_odds = base_odds
        logger.info(f"Points to Double the Odds (PDO): {self.pdo}")
        logger.info(f"Base Score: {self.base_score}")
        logger.info(f"Base Odds: {self.base_odds}")
        # Factor equation: factor = PDO / ln(2)
        self.factor = self.pdo / np.log(2)
        # Offset equation: offset = Base Score - Factor * ln(Base Odds)
        self.offset = base_score - self.factor * np.log(base_odds)
        logger.info(f"Calculated factor: {self.factor:.4f}")
        logger.info(f"Calculated offset: {self.offset:.4f}")
        logger.info(f"Scorecard formula: Score = {self.offset:.2f} + {self.factor:.2f} * ln(odds)")
        logger.info("="*50)
        
        self._load_model()
    
    def _load_model(self):
        """Load trained logistic regression model from pickle"""
        try:
            model_path = os.path.join("model", "model_fit.pkl")
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    self.model_fit = pickle.load(f)
                logger.info(f"Model loaded successfully from {model_path}")
            else:
                logger.error(f"Model file not found at {model_path}")
                raise FileNotFoundError(f"Model file not found at {model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}", exc_info=True)
            raise
    
    def predict_score(self, X):
        """
        Predict credit risk score for given features
        
        Args:
            X: Features DataFrame (single row or multiple rows)
            
        Returns:
            float or pd.Series: Credit score(s) for each observation
        """
        if self.model_fit is None:
            raise ValueError("Model is not loaded. Cannot calculate scores.")
        
        try:
            # Handle single row vs multiple rows
            if isinstance(X, pd.Series):
                X = X.to_frame().T
            
            num_records = len(X)
            logger.info("="*70)
            logger.info(f"SCORECARD PREDICTION - Processing {num_records} record(s)")
            logger.info("="*70)
            
            # Check if constant needs to be added
            expected_features = self.model_fit.params.shape[0]
            current_features = X.shape[1]
            logger.info(f"\n[STEP 0] Checking feature dimensions")
            logger.info(f"Model expects: {expected_features} features")
            logger.info(f"Input has: {current_features} features")
            
            # Add constant term to match training data structure if needed
            if current_features + 1 == expected_features:
                logger.info(f"Adding constant term...")
                # Add constant column directly
                X_with_const = X.copy()
                X_with_const.insert(0, 'const', 1.0)
                logger.info(f"Constant term added. New shape: {X_with_const.shape}")
                logger.info(f"New columns: {list(X_with_const.columns)}")
            else:
                logger.info(f"No constant needed (already has {current_features} = {expected_features} - 1)")
                X_with_const = X
            
            logger.info(f"\n[STEP 1] Input features shape: {X.shape}")
            logger.info(f"Features after preparation: {X_with_const.shape}")
            logger.info(f"Features: {list(X.columns)}")
            
            if num_records == 1:
                logger.info("\nInput values:")
                for col in X.columns:
                    logger.info(f"  {col}: {X[col].iloc[0]}")
            
            # Get predicted probabilities P(default=1)
            logger.info(f"\n[STEP 2] Getting predicted probabilities from logistic regression model...")
            predictions = self.model_fit.predict(X_with_const)
            logger.info(f"Raw predictions (probabilities): {predictions.values if hasattr(predictions, 'values') else predictions}")
            
            # Avoid division by zero
            predictions = np.clip(predictions, 1e-10, 1 - 1e-10)
            logger.info(f"Clipped predictions: {predictions.values if hasattr(predictions, 'values') else predictions}")
            
            # Calculate odds: (1-p) / p = P(good) / P(bad)
            # This ensures: high score = good customer, low score = bad customer
            logger.info(f"\n[STEP 3] Calculating odds: odds = P(good) / P(bad) = (1 - P(default)) / P(default)")
            odds = (1 - predictions) / predictions
            logger.info(f"Odds: {odds.values if hasattr(odds, 'values') else odds}")
            
            # Calculate ln(odds)
            logger.info(f"\n[STEP 4] Calculating natural logarithm of odds: ln(odds)")
            ln_odds = np.log(odds)
            logger.info(f"ln(odds): {ln_odds.values if hasattr(ln_odds, 'values') else ln_odds}")
            
            # Calculate score: offset + factor * ln(odds)
            logger.info(f"\n[STEP 5] Calculating final score:")
            logger.info(f"Formula: Score = Offset + Factor * ln(odds)")
            logger.info(f"Formula: Score = {self.offset} + {self.factor} * ln(odds)")
            scores = self.offset + self.factor * ln_odds
            
            if num_records == 1:
                score_value = scores.iloc[0] if hasattr(scores, 'iloc') else scores[0]
                ln_odds_value = ln_odds.iloc[0] if hasattr(ln_odds, 'iloc') else ln_odds[0]
                logger.info(f"Score = {self.offset} + {self.factor} × {ln_odds_value:.6f}")
                logger.info(f"Score = {self.offset} + {self.factor * ln_odds_value:.6f}")
                logger.info(f"Score = {score_value:.2f}")
            
            logger.info(f"\n[RESULT] Credit scores calculated successfully")
            if num_records > 1:
                logger.info(f"Score range: {scores.min():.2f} - {scores.max():.2f}")
                logger.info(f"Score mean: {scores.mean():.2f}, std: {scores.std():.2f}")
            else:
                final_score = scores.iloc[0] if hasattr(scores, 'iloc') else scores[0]
                logger.info(f"Final Score: {final_score:.2f}")
            logger.info("="*70)
            
            # Return single value if single input
            if num_records == 1:
                return scores.iloc[0] if hasattr(scores, 'iloc') else scores[0]
            return scores
        
        except Exception as e:
            logger.error(f"Error predicting scores: {str(e)}", exc_info=True)
            raise
    

if __name__ == "__main__":
    # Read Woe Data
    X_train_woe = pd.read_csv("data/X_train_woe.csv")
    y_train = pd.read_csv("data/y_train.csv")  

    # Get random 10 records from X_train_woe for testing
    sample_records = X_train_woe.sample(10, random_state=42)

    # Get Model Score for the sample records
    scorecard = Scorecard()
    for i, record in sample_records.iterrows():
        score = scorecard.predict_score(record)
        logger.info(f"Predicted Score for sample record {i}: {score:.2f}") 