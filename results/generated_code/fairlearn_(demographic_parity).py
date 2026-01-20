import pandas as pd
import numpy as np
from typing import Any
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline

def apply_intervention(df: pd.DataFrame, sensitive_attr: str, outcome: str) -> Any:
    """
    Apply Demographic Parity fairness intervention using Fairlearn.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing features, sensitive attribute, and outcome
    sensitive_attr : str
        Name of the sensitive attribute column
    outcome : str
        Name of the binary outcome column
        
    Returns
    -------
    Any
        Trained fair model object
    """
    
    X = df.drop(columns=[sensitive_attr, outcome]).copy()
    y = df[outcome].copy()
    sensitive_features = df[sensitive_attr].copy()
    
    if sensitive_features.dtype == 'object' or sensitive_features.dtype.name == 'category':
        le = LabelEncoder()
        sensitive_features = le.fit_transform(sensitive_features)
    
    base_estimator = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(random_state=42, max_iter=1000))
    ])
    
    constraint = DemographicParity()
    
    mitigator = ExponentiatedGradient(
        estimator=base_estimator,
        constraints=constraint,
        eps=0.01,
        max_iter=50,
        eta0=2.0
    )
    
    mitigator.fit(X, y, sensitive_features=sensitive_features)
    
    return mitigator

import pandas as pd

df = pd.read_csv('data.csv')
model = apply_intervention(df, 'race_white', 'high_cost')
predictions = model.predict(df.drop(columns=['high_cost']))
