import pandas as pd
import joblib

# Load trained model
model = joblib.load('models/final_stacking_model.pkl')

# Load the dataset used during training
df = pd.read_csv('data/engineered_features.csv')

# Define target column
target_col = 'Conversion_Rate'

# Extract features and their dtypes
X_train = df.drop(columns=[target_col])
features = X_train.columns.tolist()
dtypes = X_train.dtypes.to_dict()

# Save factorization mappings for categorical columns
factorize_maps = {}
for col in X_train.select_dtypes(include="object").columns:
    codes, uniques = pd.factorize(X_train[col])
    factorize_maps[col] = {k: v for v, k in enumerate(uniques)}

print("Please input values for the following features:\n")

# Collect user input
user_data = {}
for feature in features:
    dtype = dtypes[feature]
    while True:
        val = input(f"{feature} ({dtype}): ").strip()
        try:
            if "int" in str(dtype):
                user_data[feature] = int(val)
            elif "float" in str(dtype):
                user_data[feature] = float(val)
            else:
                user_data[feature] = str(val)
            break
        except ValueError:
            print(f"Invalid input. Please enter a value matching type {dtype}.")

# Convert to DataFrame
input_df = pd.DataFrame([user_data])

# Apply original dtypes
for col in input_df.columns:
    input_df[col] = input_df[col].astype(dtypes[col])

# Encode categorical variables using saved mapping
for col in input_df.select_dtypes(include="object").columns:
    val = input_df.at[0, col]
    mapping = factorize_maps.get(col, {})
    if val in mapping:
        input_df[col] = mapping[val]
    else:
        print(f"Warning: Unknown category '{val}' for column '{col}', defaulting to -1")
        input_df[col] = -1

# Ensure correct column order
input_df = input_df[features]

# Convert all values to int64
input_df = input_df.astype('int64')

# Predict
prediction = model.predict(input_df)[0]
print(f"\n Predicted Conversion Rate: {prediction:.4f}")
