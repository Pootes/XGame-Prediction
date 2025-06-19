from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import io, base64
import matplotlib.pyplot as plt
from visualization import generate_all_visualizations
import os

app = Flask(__name__)

# Load model and data
model = joblib.load("models/final_stacking_model.pkl")
df = pd.read_csv("data/cleaned_campaign_data.csv")
target_col = "Conversion_Rate"
X_train = df.drop(columns=[target_col])
features = X_train.columns.tolist()
dtypes = X_train.dtypes.to_dict()

# Build factorization map
factorize_maps = {}
for col in X_train.select_dtypes(include="object").columns:
    codes, uniques = pd.factorize(X_train[col])
    factorize_maps[col] = {k: v for v, k in enumerate(uniques)}

@app.route("/")
def index():
    return render_template("index.html", features=features, dtypes=dtypes)

@app.route("/predict", methods=["POST"])
def predict():
    user_data = {}
    for feature in features:
        val = request.form.get(feature, "")
        dtype = str(dtypes[feature])
        try:
            if "int" in dtype:
                user_data[feature] = int(val)
            elif "float" in dtype:
                user_data[feature] = float(val)
            else:
                user_data[feature] = str(val)
        except ValueError:
            return jsonify({"error": f"Invalid input for {feature}."})

    # Convert to DataFrame
    input_df = pd.DataFrame([user_data])
    for col in input_df.columns:
        input_df[col] = input_df[col].astype(dtypes[col])

    # Factorize
    for col in input_df.select_dtypes(include="object").columns:
        val = input_df.at[0, col]
        mapping = factorize_maps.get(col, {})
        input_df[col] = mapping.get(val, -1)

    input_df = input_df[features].astype('int64')
    prediction = model.predict(input_df)[0]
    return jsonify({"prediction": f"{prediction:.4f}"})


@app.route("/visualize")
def visualize():
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

    
    images = generate_all_visualizations(df)
    return jsonify({"images": images})

if __name__ == "__main__":
    app.run(debug=True)
