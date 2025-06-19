import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import shap
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# Load datasets
datasets = {
    "Features Cleaned": pd.read_csv("data/cleaned_campaign_data.csv"),
    "Engineered Features": pd.read_csv("data/engineered_features.csv"),
    "Reduced Features": pd.read_csv("data/reduced_campaign_data.csv")
}

metrics = []

# Function: Train + Evaluate + SHAP Explain
def train_models(df, dataset_name):
    X = df.drop(columns=["Conversion_Rate"])
    y = df["Conversion_Rate"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "XGBoost": xgb.XGBRegressor(objective='reg:squarederror', learning_rate=0.1, n_estimators=200, max_depth=5),
        "LightGBM": lgb.LGBMRegressor(objective='regression', learning_rate=0.1, n_estimators=200, max_depth=5),
        "RandomForest": RandomForestRegressor(n_estimators=200, max_depth=5, random_state=42)
    }

    shap_values_dict = {}
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        # Metrics
        metrics.append({
            "Dataset": dataset_name,
            "Model": model_name,
            "R²": round(r2_score(y_test, preds), 4),
            "MSE": round(mean_squared_error(y_test, preds), 4),
            "MAE": round(mean_absolute_error(y_test, preds), 4)
        })

        # SHAP Explainer
        explainer = shap.Explainer(model, X_train, feature_names=X.columns)
        shap_values = explainer(X_test)

        # Store mean absolute shap values
        shap_importance = pd.DataFrame({
            "feature": X.columns,
            "importance": np.abs(shap_values.values).mean(axis=0)
        }).set_index("feature").sort_values(by="importance", ascending=False)

        shap_values_dict[model_name] = shap_importance

    # Combine top 15 from all models
    all_features = pd.concat([shap_values_dict[m] for m in shap_values_dict], axis=1)
    all_features.columns = list(shap_values_dict.keys())
    top_features = all_features.max(axis=1).sort_values(ascending=False).head(15).index
    combined = all_features.loc[top_features].fillna(0)

    # Plot
    combined.sort_values(by="XGBoost", ascending=True).plot.barh(figsize=(10, 8))
    plt.title(f"SHAP Feature Importance (Top 15) - {dataset_name}")
    plt.xlabel("Mean |SHAP Value|")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()

# Run all datasets
import numpy as np
for name, df in datasets.items():
    train_models(df, name)

# Print final metrics
metrics_df = pd.DataFrame(metrics)
print("\nModel Evaluation Summary:")
print(metrics_df)
