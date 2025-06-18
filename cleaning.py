import pandas as pd
import numpy as np

def clean_data(filepath):
    df = pd.read_csv(filepath)
    duplicated_data = df[df.duplicated()].sort_values('Campaign_ID', ascending=True)
    # Display duplicated data
    if duplicated_data.empty:
        print("No duplicated data found.")
    else:
        print("Duplicated data found:")
        print(duplicated_data)

    # Data Cleaning
    df.columns = df.columns.str.strip()
    # Convert Campaign_ID column from float to object
    df['Campaign_ID'] = df['Campaign_ID'].astype(str)

    # Remove $ from number in Acquisition_Cost column
    df['Acquisition_Cost'] = df['Acquisition_Cost'].str.replace('$', '', regex=False)

    # Remove , from number in Acquisition_Cost column
    df['Acquisition_Cost'] = df['Acquisition_Cost'].str.replace(',', '', regex=False)

    # Convert Acquisition_Cost column from object to float
    df['Acquisition_Cost'] = df['Acquisition_Cost'].astype(float)

    # Convert into a correct datatype
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')

    # Clean up infinities and NaNs
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    # Drop unused columns
    df = df.drop(columns=['Campaign_ID'])
    
    return df
