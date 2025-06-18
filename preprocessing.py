import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def preprocess_data(df, save_engineered=True):
    df = df.copy()

    # Label encode categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    le = LabelEncoder()
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])

    # --- Feature Engineering ---
    df_engineered = df.copy()
    df_engineered['Clicks_Per_Impression'] = df_engineered['Clicks'] / df_engineered['Impressions'].replace(0, np.nan)
    df_engineered['Cost_Per_Click'] = df_engineered['Acquisition_Cost'] / df_engineered['Clicks'].replace(0, np.nan)
    df_engineered['Cost_Per_Engagement'] = df_engineered['Acquisition_Cost'] / df_engineered['Engagement_Score'].replace(0, np.nan)
    df_engineered['ROI_Per_Cost'] = df_engineered['ROI'] / df_engineered['Acquisition_Cost'].replace(0, np.nan)
    df_engineered['Efficiency_Score'] = (df_engineered['Engagement_Score'] * df_engineered['ROI']) / df_engineered['Acquisition_Cost'].replace(0, np.nan)

    # Optional: save to a different CSV for comparison
    if save_engineered:
        df_engineered.to_csv("data/engineered_features.csv", index=False)

    return df, df_engineered  # original_features, engineered_features
