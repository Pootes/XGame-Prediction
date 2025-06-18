import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import optuna
import joblib
from sklearn.model_selection import train_test_split

# Load preprocessed dataset with engineered features
engineered_df = pd.read_csv("data/engineered_dataset.csv")
X = engineered_df.drop("Conversion_Rate", axis=1)
y = engineered_df["Conversion_Rate"]

# Define K-Fold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Objective function for Optuna hyperparameter tuning
def train_models(trial):
    rf_params = {
        "n_estimators": trial.suggest_int("rf_n_estimators", 100, 500),
        "max_depth": trial.suggest_int("rf_max_depth", 3, 20),
        "min_samples_split": trial.suggest_int("rf_min_samples_split", 2, 10),
    }
    xgb_params = {
        "n_estimators": trial.suggest_int("xgb_n_estimators", 100, 500),
        "max_depth": trial.suggest_int("xgb_max_depth", 3, 20),
        "learning_rate": trial.suggest_float("xgb_learning_rate", 0.01, 0.3),
    }
    lgbm_params = {
        "n_estimators": trial.suggest_int("lgbm_n_estimators", 100, 500),
        "max_depth": trial.suggest_int("lgbm_max_depth", 3, 20),
        "learning_rate": trial.suggest_float("lgbm_learning_rate", 0.01, 0.3),
    }

    rf = RandomForestRegressor(**rf_params, random_state=42)
    xgb = XGBRegressor(**xgb_params, random_state=42)
    lgbm = LGBMRegressor(**lgbm_params, random_state=42)

    estimators = [
        ("rf", rf),
        ("xgb", xgb),
        ("lgbm", lgbm)
    ]

    stack_model = StackingRegressor(
        estimators=estimators,
        final_estimator=Ridge(alpha=1.0)
    )

    # Cross-validation RMSE
    scores = cross_val_score(stack_model, X, y, scoring="neg_root_mean_squared_error", cv=kf)
    return -1 * scores.mean()

# Run Optuna optimization
study = optuna.create_study(direction="minimize")
study.optimize(train_models, n_trials=50)

# Get best parameters and retrain final model
best_params = study.best_params

best_rf = RandomForestRegressor(
    n_estimators=best_params["rf_n_estimators"],
    max_depth=best_params["rf_max_depth"],
    min_samples_split=best_params["rf_min_samples_split"],
    random_state=42
)

best_xgb = XGBRegressor(
    n_estimators=best_params["xgb_n_estimators"],
    max_depth=best_params["xgb_max_depth"],
    learning_rate=best_params["xgb_learning_rate"],
    random_state=42
)

best_lgbm = LGBMRegressor(
    n_estimators=best_params["lgbm_n_estimators"],
    max_depth=best_params["lgbm_max_depth"],
    learning_rate=best_params["lgbm_learning_rate"],
    random_state=42
)

final_stack = StackingRegressor(
    estimators=[
        ("rf", best_rf),
        ("xgb", best_xgb),
        ("lgbm", best_lgbm)
    ],
    final_estimator=Ridge(alpha=1.0)
)

# Fit final model
final_stack.fit(X, y)

# Save model
joblib.dump(final_stack, "models/final_stacking_model.pkl")
print("Final model trained and saved.")

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# Fit final model on training data
final_stack.fit(X_train, y_train)

# Predict on test set
y_pred = final_stack.predict(X_test)

# Evaluate performance
mse = mean_squared_error(y_test, y_pred)
print(f"Test MSE: {mse:.4f}")

# Save the model
joblib.dump(final_stack, "models/final_stacking_model.pkl")
print("Final model trained and saved.")
