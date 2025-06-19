import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load datasets
clean = pd.read_csv("data/cleaned_campaign_data.csv")
eng = pd.read_csv("data/engineered_features.csv")
reduced = pd.read_csv("data/reduced_campaign_data.csv")

# Evaluation + training function
def train_and_evaluate(df, name):
    X = df.drop(columns=["Conversion_Rate"])
    y = df["Conversion_Rate"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = lgb.LGBMRegressor(
        objective='regression',
        metric='rmse',
        boosting_type='gbdt',
        learning_rate=0.1,
        n_estimators=200,
        max_depth=5,        
    )


    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Metrics
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    # Feature importance plot
    importance = pd.Series(model.feature_importances_, index=X.columns)
    top_features = importance.sort_values(ascending=False).head(15)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_features.values, y=top_features.index)
    plt.title(f"Top 15 Feature Importances - {name}")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()

    return {
        "Dataset": name,
        "R²": round(r2, 4),
        "MSE": round(mse, 4),
        "MAE": round(mae, 4)
    }

# Run on all datasets
results = [
    train_and_evaluate(clean, "Features Cleaned"),
    train_and_evaluate(eng, "Engineered Features"),
    train_and_evaluate(reduced, "Reduced Features")
]

# Display summary
results_df = pd.DataFrame(results)
print("\n Model Evaluation Summary:")
print(results_df)
