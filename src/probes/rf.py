# Random Forest Classifier Implementation

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
import numpy as np


class RFProbe:
    """Random Forest probe with 5-fold cross-validation for hyperparameter selection."""
    
    def __init__(self, param_grid: dict = None, cv_folds: int = 5, random_state: int = 42):
        """
        Initialize RF probe.
        
        Args:
            param_grid: Dictionary of hyperparameters to search over
            cv_folds: Number of cross-validation folds
            random_state: Random state for reproducibility
        """
        if param_grid is None:
            self.param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 10, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'class_weight': [None, 'balanced']
            }
        else:
            self.param_grid = param_grid
            
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.best_estimator_ = None
        self.best_params_ = None
        self.cv_results_ = None
        
    def fit(self, X: np.ndarray, y: np.ndarray, selection_metric: str = 'auprc', n_jobs: int = -1) -> 'RFProbe':
        """
        Fit RF probe with hyperparameter search using cross-validation.
        
        Args:
            X: Training features
            y: Training labels
            selection_metric: Metric to use for hyperparameter selection ('auprc' or 'auroc')
            
        Returns:
            self
        """
        # Set up cross-validation
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        
        # Set up scoring metric
        if selection_metric == 'auprc':
            scoring = 'average_precision'
        elif selection_metric == 'auroc':
            scoring = 'roc_auc'
        else:
            raise ValueError(f"Unknown selection metric: {selection_metric}")
        
        # Initialize base classifier
        rf = RandomForestClassifier(random_state=self.random_state)
        
        # Perform grid search
        grid_search = GridSearchCV(
            estimator=rf,
            param_grid=self.param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=1
        )
        
        # Fit the grid search
        grid_search.fit(X, y)
        
        # Store results
        self.best_estimator_ = grid_search.best_estimator_
        self.best_params_ = grid_search.best_params_
        self.cv_results_ = grid_search.cv_results_
        
        # Print best parameters and score
        best_score = grid_search.best_score_
        print(f"\nBest hyperparameters: {self.best_params_}")
        print(f"Best CV {selection_metric}: {best_score:.4f}")
        
        return self
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities for positive class.
        
        Args:
            X: Test features
            
        Returns:
            Predicted probabilities for positive class
        """
        if self.best_estimator_ is None:
            raise ValueError("Model must be fitted before making predictions")
            
        return self.best_estimator_.predict_proba(X)[:, 1]
    