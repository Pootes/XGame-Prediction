from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd

def preprocess_data(df):
    df = df.copy()

    # --- Feature Engineering ---
    df['Clicks_Per_Impression'] = df['Clicks'] / df['Impressions'].replace(0, np.nan)
    df['Cost_Per_Click'] = df['Acquisition_Cost'] / df['Clicks'].replace(0, np.nan)
    df['Cost_Per_Engagement'] = df['Acquisition_Cost'] / df['Engagement_Score'].replace(0, np.nan)
    df['ROI_Per_Cost'] = df['ROI'] / df['Acquisition_Cost'].replace(0, np.nan)
    df['Efficiency_Score'] = (df['Engagement_Score'] * df['ROI']) / df['Acquisition_Cost'].replace(0, np.nan)
    
    # Label encode all categorical (object) columns
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    df.fillna(0, inplace=True)
    df.to_csv("data/engineered_features.csv", index=False)

    return df
