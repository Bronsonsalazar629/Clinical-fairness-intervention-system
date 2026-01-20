import pandas as pd
import numpy as np
from typing import Any
from aif360.algorithms.preprocessing import Reweighing
from aif360.datasets import StandardDataset
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def apply_intervention(df: pd.DataFrame, sensitive_attr: str, outcome: str) -> Any:
    """
    Apply AIF360 Reweighing preprocessing intervention to mitigate bias.
    
    Args:
        df: Input DataFrame containing features, sensitive attribute, and outcome
        sensitive_attr: Name of the sensitive attribute column
        outcome: Name of the binary outcome/target column (0/1)
    
    Returns:
        Trained logistic regression model with reweighted samples
    """
    data = df.copy()
    
    feature_cols = [col for col in data.columns 
                   if col not in [sensitive_attr, outcome]]
    
    X = data[feature_cols].values
    y = data[outcome].values
    sensitive = data[sensitive_attr].values
    
    dataset = StandardDataset(
        df=data,
        label_name=outcome,
        favorable_classes=[1],
        protected_attribute_names=[sensitive_attr],
        privileged_classes=[[1]]
    )
    
    RW = Reweighing(
        unprivileged_groups=[{sensitive_attr: 0}],
        privileged_groups=[{sensitive_attr: 1}]
    )
    dataset_transformed = RW.fit_transform(dataset)
    
    X_transformed = dataset_transformed.features
    y_transformed = dataset_transformed.labels.ravel()
    sample_weights = dataset_transformed.instance_weights
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_transformed)
    
    model = LogisticRegression(
        random_state=42,
        max_iter=1000,
        class_weight='balanced'
    )
    model.fit(X_scaled, y_transformed, sample_weight=sample_weights)
    
    return model

import pandas as pd

df = pd.read_csv('data.csv')
model = apply_intervention(df, 'race_white', 'high_cost')
predictions = model.predict(df.drop(columns=['high_cost']))
