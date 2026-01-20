import pandas as pd
import numpy as np
from typing import Any
from sklearn.linear_model import LogisticRegression
from fairlearn.reductions import EqualizedOdds
from fairlearn.reductions import GridSearch

def apply_intervention(df: pd.DataFrame, sensitive_attr: str, outcome: str) -> Any:
    """
    Apply Equalized Odds fairness intervention using Fairlearn.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing features, sensitive attribute, and outcome
    sensitive_attr : str
        Name of the sensitive attribute column (e.g., 'gender', 'race')
    outcome : str
        Name of the binary outcome/target column
        
    Returns
    -------
    Any
        Trained model object implementing Equalized Odds fairness constraint
    """
    X = df.drop(columns=[sensitive_attr, outcome]).values
    y = df[outcome].values
    sensitive_features = df[sensitive_attr].values
    
    base_estimator = LogisticRegression(solver='liblinear', random_state=42)
    
    constraint = EqualizedOdds()
    
    mitigator = GridSearch(
        estimator=base_estimator,
        constraints=constraint,
        grid_size=50,
        grid_limit=2.0
    )
    
    mitigator.fit(X, y, sensitive_features=sensitive_features)
    
    return mitigator

import pandas as pd

df = pd.read_csv('data.csv')
model = apply_intervention(df, 'race_white', 'high_cost')
predictions = model.predict(df.drop(columns=['high_cost']))
