import pandas as pd
import numpy as np
import os
import joblib
import optuna
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from inputimeout import inputimeout, TimeoutOccurred


# Config
TARGET_COL = "Conversion_Rate"
N_TRIALS = 50
TEST_SIZE = 0.2
RANDOM_STATE = 42
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

STUDY_PATH = os.path.join(MODEL_DIR, "optuna_study.pkl")
MODEL_PATH = os.path.join(MODEL_DIR, "final_stacking_model.pkl")

# Load and Preprocess Data 
engineered_df = pd.read_csv("data/cleaned_campaign_data.csv")

# Encode categorical variables
for col in engineered_df.select_dtypes(include=["object"]):
    engineered_df[col] = LabelEncoder().fit_transform(engineered_df[col].astype(str))

X = engineered_df.drop(TARGET_COL, axis=1)
y = engineered_df[TARGET_COL]

kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# Model Factory
def create_model(xgb_params, lgbm_params):
    xgb = XGBRegressor(**xgb_params, random_state=RANDOM_STATE, verbosity=0)
    lgbm = LGBMRegressor(**lgbm_params, random_state=RANDOM_STATE)
    return StackingRegressor(
        estimators=[("xgb", xgb), ("lgbm", lgbm)],
        final_estimator=Ridge(alpha=1.0),
        n_jobs=-1
    )

# Optuna Objective
def train_models(trial):
    xgb_params = {
        "n_estimators": trial.suggest_int("xgb_n_estimators", 50, 300),
        "max_depth": trial.suggest_int("xgb_max_depth", 4, 10),
        "learning_rate": trial.suggest_float("xgb_learning_rate", 0.05, 0.2),
        "n_jobs": -1
    }
    lgbm_params = {
        "n_estimators": trial.suggest_int("lgbm_n_estimators", 50, 300),
        "max_depth": trial.suggest_int("lgbm_max_depth", 4, 10),
        "learning_rate": trial.suggest_float("lgbm_learning_rate", 0.05, 0.2),
        "n_jobs": -1
    }

    model = create_model(xgb_params, lgbm_params)
    scores = cross_val_score(model, X, y, scoring="neg_root_mean_squared_error", cv=kf, n_jobs=-1)
    return -scores.mean()

# User Choice
load_existing = False
if os.path.exists(STUDY_PATH) and os.path.exists(MODEL_PATH):
    try:
        user_input = inputimeout(prompt="Existing Optuna study and model found. Load them? (y/n): ", timeout=30).strip().lower()
        if user_input == "y":
            load_existing = True
    except TimeoutOccurred:
        print("\n[Timed out after 30s] Proceeding with new Optuna study")


if load_existing:
    print("Loading existing study and model...")
    study = joblib.load(STUDY_PATH)
    final_model = joblib.load(MODEL_PATH)
else:
    print("Running new Optuna optimization...")
    study = optuna.create_study(direction="minimize")
    study.optimize(train_models, n_trials=N_TRIALS)

    print("Best parameters found:", study.best_params)
    print("Best RMSE from cross-validation:", study.best_value)

    # Extract best params
    best_xgb_params = {
        k.replace("xgb_", ""): v for k, v in study.best_params.items() if k.startswith("xgb_")
    }
    best_lgbm_params = {
        k.replace("lgbm_", ""): v for k, v in study.best_params.items() if k.startswith("lgbm_")
    }

    # Train final model
    final_model = create_model(best_xgb_params, best_lgbm_params)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    final_model.fit(X_train, y_train)

    # Save
    joblib.dump(final_model, MODEL_PATH)
    joblib.dump(study, STUDY_PATH)
    print("Final model and Optuna study saved to 'models/'")

# Evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
y_pred = final_model.predict(X_test)

rmse = root_mean_squared_error(y_test, y_pred, squared=False)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nTest Evaluation:\n - RMSE: {rmse:.4f}\n - MAE: {mae:.4f}\n - R²: {r2:.4f}")
