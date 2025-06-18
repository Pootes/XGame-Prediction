from cleaning import clean_data
from model_training import train_models
from visualization import plot_correlation_matrix
from preprocessing import preprocess_data
import optuna


if __name__ == "__main__":
    df = clean_data("data/marketing_campaign_dataset 2.csv")
    df.info()
    df['Company'].unique()
    df['Campaign_Type'].unique()
    df['Target_Audience'].unique()
    df['Duration'].unique()
    df['Channel_Used'].unique()
    df['Location'].unique()
    df['Customer_Segment'].unique()
    df.describe()
    plot_correlation_matrix(df)
    preprocess_data(df)
    train_models(trial=10)