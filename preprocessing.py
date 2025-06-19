import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# === STEP 1: LOAD & CLEAN ===
df = pd.read_csv("data/marketing_campaign_dataset 2.csv")

# --- Cleaning ---
drop_cols = ['Campaign_ID']
df.drop(columns=drop_cols, inplace=True)
df['Acquisition_Cost'] = df['Acquisition_Cost'].replace('[\$,]', '', regex=True).astype(float)
df['Duration'] = df['Duration'].str.extract('(\d+)').astype(int)
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df['Campaign_Weekday'] = df['Date'].dt.weekday
df['Campaign_Month'] = df['Date'].dt.month
df['Campaign_Quarter'] = df['Date'].dt.quarter
df['Is_Weekend'] = df['Campaign_Weekday'].isin([5, 6]).astype(int)
df = df.drop(columns=['Date'])

# Use this cleaned version as the base for all exports
df_cleaned = df.copy()

# === STEP 2: RAW CLEANED DATA (Label Encoded) ===

for col in df_cleaned.select_dtypes(include=['object']).columns:
    df_cleaned[col] = LabelEncoder().fit_transform(df_cleaned[col].astype(str))

df_cleaned.fillna(0, inplace=True)
df_cleaned.to_csv("data/cleaned_campaign_data.csv", index=False)

# === STEP 3: ENGINEERED FEATURES (Label Encoded) ===
df_eng = df_cleaned.copy()

# Derived post-campaign metrics
df_eng['Clicks_Per_Impression'] = df_eng['Clicks'] / df_eng['Impressions'].replace(0, np.nan)
df_eng['Cost_Per_Click'] = df_eng['Acquisition_Cost'] / df_eng['Clicks'].replace(0, np.nan)
df_eng['Cost_Per_Engagement'] = df_eng['Acquisition_Cost'] / df_eng['Engagement_Score'].replace(0, np.nan)
df_eng['ROI_Per_Cost'] = df_eng['ROI'] / df_eng['Acquisition_Cost'].replace(0, np.nan)
df_eng['Efficiency_Score'] = (df_eng['Engagement_Score'] * df_eng['ROI']) / df_eng['Acquisition_Cost'].replace(0, np.nan)

# Apply same encoding strategy
for col in df_eng.select_dtypes(include=['object']).columns:
    df_eng[col] = LabelEncoder().fit_transform(df_eng[col].astype(str))

df_eng.fillna(0, inplace=True)
df_eng.to_csv("data/engineered_features.csv", index=False)

# === STEP 4: REDUCED DATASET (Label Encoded) ===
df_reduced = df_cleaned.copy()

# Drop post-campaign columns
drop_cols = ['ROI', 'Acquisition_Cost', 'Clicks', 'Impressions', 'Engagement_Score']
df_reduced.drop(columns=drop_cols, inplace=True)

# Apply label encoding to categorical features
for col in df_reduced.select_dtypes(include=['object']).columns:
    df_reduced[col] = LabelEncoder().fit_transform(df_reduced[col].astype(str))

df_reduced.fillna(0, inplace=True)
df_reduced.to_csv("data/reduced_campaign_data.csv", index=False)

print(" All datasets saved")

