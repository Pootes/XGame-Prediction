import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import optuna
import joblib
import os

# Load dataset
engineered_df = pd.read_csv("data/engineered_features.csv")

# Encode categorical variables
for col in engineered_df.select_dtypes(include=['object']).columns:
    engineered_df[col] = LabelEncoder().fit_transform(engineered_df[col].astype(str))

X = engineered_df.drop("Conversion_Rate", axis=1)
y = engineered_df["Conversion_Rate"]

# Define K-Fold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Objective function for Optuna tuning
def train_models(trial):
    xgb_params = {
        "n_estimators": trial.suggest_int("xgb_n_estimators", 50, 150),
        "max_depth": trial.suggest_int("xgb_max_depth", 3, 6),
        "learning_rate": trial.suggest_float("xgb_learning_rate", 0.1, 0.3),
        "n_jobs": -1
    }
    lgbm_params = {
        "n_estimators": trial.suggest_int("lgbm_n_estimators", 50, 150),
        "max_depth": trial.suggest_int("lgbm_max_depth", 3, 6),
        "learning_rate": trial.suggest_float("lgbm_learning_rate", 0.1, 0.3),
        "n_jobs": -1
    }

    xgb = XGBRegressor(**xgb_params, random_state=42, verbosity=0)
    lgbm = LGBMRegressor(**lgbm_params, random_state=42)

    stack_model = StackingRegressor(
        estimators=[("xgb", xgb), ("lgbm", lgbm)],
        final_estimator=Ridge(alpha=1.0),
        n_jobs=-1
    )

    scores = cross_val_score(stack_model, X, y, scoring="neg_root_mean_squared_error", cv=kf, n_jobs=-1)
    return -scores.mean()

# Run Optuna study
study = optuna.create_study(direction="minimize")
study.optimize(train_models, n_trials=50)

# Train final model with best parameters
best_params = study.best_params

best_xgb = XGBRegressor(
    n_estimators=best_params["xgb_n_estimators"],
    max_depth=best_params["xgb_max_depth"],
    learning_rate=best_params["xgb_learning_rate"],
    n_jobs=-1,
    random_state=42,
    verbosity=0
)

best_lgbm = LGBMRegressor(
    n_estimators=best_params["lgbm_n_estimators"],
    max_depth=best_params["lgbm_max_depth"],
    learning_rate=best_params["lgbm_learning_rate"],
    n_jobs=-1,
    random_state=42
)

final_stack = StackingRegressor(
    estimators=[("xgb", best_xgb), ("lgbm", best_lgbm)],
    final_estimator=Ridge(alpha=1.0),
    n_jobs=-1
)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit and evaluate
final_stack.fit(X_train, y_train)
y_pred = final_stack.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Test MSE: {mse:.4f}")
study = optuna.create_study(direction="minimize")
study.optimize(train_models, n_trials=50)
print("Best parameters found: ", study.best_params)
print("Best RMSE: ", study.best_value)
print("Training completed successfully.")

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(final_stack, "models/final_stacking_model.pkl")
print("Final model trained and saved.")
