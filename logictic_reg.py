import numpy as np
import pandas as pd
import statsmodels.api as sm
import json
import pickle
import os
from logging_config import get_logger

logger = get_logger("logistic_regression", "logistic_regression.log", True, "w")
MODEL_DIR = "model"


class LogisticRegression:
    def __init__(self):
        self.model = None
        self.model_fit = None
        logger.info("LogisticRegression instance initialized")
        
        # Create model directory if not exists
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)
            logger.info(f"Created model directory: {MODEL_DIR}")
    
    def fit(self, X, y):
        """
        Train Logistic Regression model
        
        Args:
            X: Features DataFrame
            y: Target Series
            
        Returns:
            LogisticRegressionResult: Result object containing model and summary data
        """
        try:
            logger.info("="*50)
            logger.info("Starting Logistic Regression model training")
            logger.info(f"Input data shape - Features: {X.shape}, Target: {y.shape}")
            logger.info(f"Number of features: {X.shape[1]}")
            logger.info(f"Number of observations: {X.shape[0]}")
            logger.info(f"Target distribution - Class 0: {(y==0).sum()}, Class 1: {(y==1).sum()}")
            
            # Add constant term
            logger.info("Adding constant term to features")
            X_with_const = sm.add_constant(X)
            
            # Create Logit model
            logger.info("Creating Logit model")
            logger.info(f"Target Unique values: {y.unique()}")
            self.model = sm.Logit(y, X_with_const)
            
            # Fit the model
            logger.info("Fitting model...")
            self.model_fit = self.model.fit(disp=0)
            
            logger.info(f"Model training completed successfully")
            logger.info(f"Log-Likelihood: {self.model_fit.llf:.4f}")
            logger.info(f"AIC: {self.model_fit.aic:.4f}")
            logger.info(f"BIC: {self.model_fit.bic:.4f}")
            
            # Extract summary data
            logger.info("Extracting summary statistics")
            summary_dict = self._extract_summary()
            
            logger.info("Logistic Regression model ready for predictions")
            
            # Save results to model directory
            self._save_results(summary_dict)
            logger.info("="*50)
            
            return summary_dict
        
        except Exception as e:
            logger.error(f"Error during model training: {str(e)}", exc_info=True)
            raise
    
    def _save_results(self, summary_dict):
        """Save model and results to model directory"""
        try:
            logger.info("Saving model results to model directory")
            
            # Save model fit object as pickle
            model_fit_path = os.path.join(MODEL_DIR, "model_fit.pkl")
            with open(model_fit_path, 'wb') as f:
                pickle.dump(self.model_fit, f)
            logger.info(f"Model fit saved to: {model_fit_path}")
            
            # Save summary as JSON
            summary_path = os.path.join(MODEL_DIR, "summary.json")
            with open(summary_path, 'w') as f:
                json.dump(summary_dict, f, indent=4, default=str)
            logger.info(f"Summary saved to: {summary_path}")
            
            # Save model info as JSON
            model_info = {
                'model_type': 'Logistic Regression (Statsmodels)',
                'model_info': summary_dict['model_info'],
                'fit_quality': summary_dict['fit_quality'],
                'coefficients_summary': summary_dict['coefficients_summary']
            }
            model_info_path = os.path.join(MODEL_DIR, "model_info.json")
            with open(model_info_path, 'w') as f:
                json.dump(model_info, f, indent=4, default=str)
            logger.info(f"Model info saved to: {model_info_path}")
            
            logger.info("All model results saved successfully")
        
        except Exception as e:
            logger.error(f"Error saving model results: {str(e)}", exc_info=True)
            raise
    
    def _extract_summary(self):
        """Extract summary data in a nicely formatted structure"""
        if self.model_fit is None:
            logger.error("Model must be fitted first")
            raise ValueError("Model must be fitted first")
        
        logger.debug("Generating model summary")
        summary = self.model_fit.summary()

        logger.info("-"*50)
        logger.info("MODEL SUMMARY:")
        logger.info(summary)
        logger.info("-"*50)
        
        # Extract coefficients and statistics
        logger.debug("Extracting coefficients and statistics")
        params_table = self.model_fit.summary2().tables[1]
        
        # Extract model fit quality metrics with descriptions
        logger.debug("Extracting model fit quality metrics")
        fit_quality_summary = self._get_fit_quality_summary(self.model_fit)
        
        # Extract linear equation
        logger.debug("Extracting linear equation")
        linear_equation = self._generate_linear_equation(self.model_fit)
        
        summary_dict = {
            'model_info': {
                'dependent_var': str(self.model_fit.model.endog_names),
                'num_obs': int(self.model_fit.nobs),
                'df_resid': int(self.model_fit.df_resid),
                'df_model': int(self.model_fit.df_model),
            },
            'fit_quality': fit_quality_summary,
            'linear_equation': linear_equation,
            'coefficients': params_table.to_dict(),
            'coefficients_summary': self._get_coeff_summary(self.model_fit),
            'model_summary': str(summary),
        }
        
        logger.info(f"Summary extraction completed - {len(params_table)} coefficients extracted")
        return summary_dict
    
    def _get_fit_quality_summary(self, model_fit):
        """
        Extract fit quality metrics with descriptions
        Provides interpretation of model fit metrics similar to coefficient summary
        """
        logger.debug("Calculating fit quality summary with descriptions")
        
        llf_value = float(model_fit.llf)
        aic_value = float(model_fit.aic)
        bic_value = float(model_fit.bic)
        pseudo_rsquared = float(model_fit.prsquared)
        
        # Generate quality assessment based on pseudo R-squared
        if pseudo_rsquared > 0.5:
            quality_assessment = "Excellent - Model explains substantial variation in target"
        elif pseudo_rsquared > 0.3:
            quality_assessment = "Good - Model explains moderate variation in target"
        elif pseudo_rsquared > 0.1:
            quality_assessment = "Acceptable - Model explains some variation in target"
        else:
            quality_assessment = "Poor - Model explains little variation in target"
        
        fit_quality_summary = {
            'llf': {
                'value': llf_value,
                'metric': 'Log-Likelihood Function',
                'description': 'Measures the fit of the model. Lower (more negative) values indicate worse fit.',
                'range': 'Typically negative (unbounded below). Relative metric - compare between models.',
                'interpretation': f'Current value {llf_value:.2f} - Used for likelihood ratio tests'
            },
            'aic': {
                'value': aic_value,
                'metric': 'Akaike Information Criterion',
                'description': 'Model selection criterion that balances fit and complexity. Lower is better.',
                'range': 'Positive values (can be negative). Relative metric - compare between models. Lower indicates better fit.',
                'acceptable_interpretation': 'No absolute threshold - use for model comparison only',
                'interpretation': f'Current value {aic_value:.2f} - Compare with other models to select best fit'
            },
            'bic': {
                'value': bic_value,
                'metric': 'Bayesian Information Criterion',
                'description': 'Similar to AIC but penalizes model complexity more heavily. Lower is better.',
                'range': 'Positive values (can be negative). Relative metric - compare between models. Penalizes complexity more than AIC.',
                'acceptable_interpretation': 'No absolute threshold - use for model comparison only',
                'interpretation': f'Current value {bic_value:.2f} - Preferred when sample size is large'
            },
            'log_likelihood': {
                'value': llf_value,
                'metric': 'Log-Likelihood',
                'description': 'Probability of observing the training data given the model parameters.',
                'range': 'Typically negative (unbounded below). Higher (less negative) values indicate better fit.',
                'interpretation': f'Current value {llf_value:.2f} - Basis for likelihood ratio tests and model comparison'
            },
            'pseudo_r_squared': {
                'value': pseudo_rsquared,
                'metric': 'Pseudo R-squared (McFadden)',
                'description': 'Indicates proportion of variation in target explained by model (0 to 1 scale).',
                'range': '0.0 to 1.0 - Absolute scale metric',
                'range_interpretation': {
                    '0.0 - 0.1': 'Poor fit',
                    '0.1 - 0.3': 'Acceptable fit',
                    '0.3 - 0.5': 'Good fit',
                    '0.5 - 1.0': 'Excellent fit'
                },
                'interpretation': f'Current value {pseudo_rsquared:.4f} ({pseudo_rsquared*100:.2f}%) - {quality_assessment}'
            }
        }
        
        logger.info(f"Fit quality assessment: {quality_assessment}")
        return fit_quality_summary
    
    def _get_coeff_summary(self, model_fit):
        """Summary of coefficients with statistics"""
        logger.debug("Calculating coefficient summary statistics")
        params = model_fit.params
        pvalues = model_fit.pvalues
        conf_int = model_fit.conf_int()
        
        coeff_summary = {}
        significant_count = 0
        
        for param_name in params.index:
            is_significant = pvalues[param_name] < 0.05
            if is_significant:
                significant_count += 1
            
            # Generate interpretation comment based on p-value
            p_val = float(pvalues[param_name])
            if p_val < 0.001:
                interpretation = "Highly significant - Strong effect on target variable"
            elif p_val < 0.01:
                interpretation = "Very significant - Strong effect on target variable"
            elif p_val < 0.05:
                interpretation = "Significant - Moderate effect on target variable"
            else:
                interpretation = "Not significant - No meaningful effect on target variable"
            
            # Generate odds ratio interpretation
            odds_ratio = float(np.exp(params[param_name]))
            if odds_ratio > 1:
                odds_interpretation = f"Unit increase increases odds by {(odds_ratio - 1) * 100:.2f}%"
            else:
                odds_interpretation = f"Unit increase decreases odds by {(1 - odds_ratio) * 100:.2f}%"
                
            coeff_summary[param_name] = {
                'coefficient': float(params[param_name]),
                'p_value': float(pvalues[param_name]),
                'ci_lower': float(conf_int.loc[param_name, 0]),
                'ci_upper': float(conf_int.loc[param_name, 1]),
                'odds_ratio': odds_ratio,
                'significant': 'Yes' if is_significant else 'No',
                'interpretation': interpretation,
                'odds_interpretation': odds_interpretation
            }
        
        logger.info(f"Coefficient analysis: {significant_count}/{len(params.index)} significant at p<0.05")
        return coeff_summary
    
    def _generate_linear_equation(self, model_fit):
        """
        Generate linear regression equation string from coefficients
        Formula: z = beta0 + beta1*x1 + beta2*x2 + ... (logit scale)
        Returns: Formatted equation string
        """
        logger.debug("Generating linear equation from coefficients")
        params = model_fit.params
        
        # Start with constant term
        equation_parts = []
        
        for param_name in params.index:
            coeff = float(params[param_name])
            
            if param_name == 'const':
                # Constant term
                equation_parts.append(f"{coeff:.6f}")
            else:
                # Feature terms
                if coeff >= 0:
                    equation_parts.append(f"+ {coeff:.6f}*{param_name}")
                else:
                    equation_parts.append(f"- {abs(coeff):.6f}*{param_name}")
        
        linear_equation = "z = " + " ".join(equation_parts)
        
        # Generate alternative interpretable form (rounded to 4 decimals)
        rounded_parts = []
        for param_name in params.index:
            coeff = float(params[param_name])
            
            if param_name == 'const':
                rounded_parts.append(f"{coeff:.4f}")
            else:
                if coeff >= 0:
                    rounded_parts.append(f"+ {coeff:.4f}*{param_name}")
                else:
                    rounded_parts.append(f"- {abs(coeff):.4f}*{param_name}")
        
        linear_equation_rounded = "z = " + " ".join(rounded_parts)
        
        logger.info(f"Linear equation generated successfully")
        
        return {
            'equation_full_precision': linear_equation,
            'equation_rounded': linear_equation_rounded,
            'description': 'Logit linear equation (z). Probability = exp(z) / (1 + exp(z))',
            'note': 'Each feature coefficient shows effect on log-odds scale'
        }
    
    def predict(self, X, include_const=True):
        """
        Make predictions
        
        Args:
            X: Features DataFrame
            include_const: Whether to include constant term
            
        Returns:
            Prediction results
        """
        if self.model_fit is None:
            logger.error("Cannot make predictions - Model must be fitted first")
            raise ValueError("Model must be fitted first")
        
        logger.info(f"Making predictions on {X.shape[0]} observations")
        
        if include_const:
            logger.debug("Adding constant term to prediction features")
            X_pred = sm.add_constant(X)
        else:
            X_pred = X
        
        logger.debug("Generating model predictions")
        predictions = self.model_fit.predict(X_pred)
        
        logger.info(f"Predictions completed - Mean probability: {predictions.mean():.4f}")
        return predictions
    
    def get_summary(self):
        """Return model summary"""
        if self.model_fit is None:
            logger.error("Cannot get summary - Model must be fitted first")
            raise ValueError("Model must be fitted first")
        
        logger.info("Returning model summary")
        return self.model_fit.summary()

if __name__ == "__main__":

    X_train_woe = pd.read_csv("data/X_train_woe.csv")
    y_train = pd.read_csv("data/y_train.csv").squeeze()  # Convert to Series

    log_reg = LogisticRegression()
    result = log_reg.fit(X_train_woe, y_train)