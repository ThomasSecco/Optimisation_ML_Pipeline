import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(filepath):
    return pd.read_csv(filepath, sep=";")

def preprocess_data(df):
    # Handle missing values and normalize features
    df.fillna(method='ffill', inplace=True)
    X = df.drop('quality', axis=1)
    y = df['quality']
    return X, y

def save_processed_data(X, y):
    processed = pd.concat([X, y], axis=1)
    processed.to_csv("data/processed_data.csv", index=False)

if __name__ == "__main__":
    df = load_data("data/winequality-red.csv")
    X, y = preprocess_data(df)
    save_processed_data(X, y)
    print("Preprocessing complete. Data saved.")
