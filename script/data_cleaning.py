import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(filepath):
    return pd.read_csv(filepath)

def handle_missing_values(df):
    df['Sales'].fillna(df['Sales'].mean(), inplace=True)
    return df

def remove_duplicates(df):
    df.drop_duplicates(inplace=True)
    return df

def handle_outliers(df):
    sales_mean = df['Sales'].mean()
    sales_std = df['Sales'].std()
    df = df[(df['Sales'] > (sales_mean - 3 * sales_std)) & (df['Sales'] < (sales_mean + 3 * sales_std))]
    return df

def encode_categorical_variables(df):
    return pd.get_dummies(df, columns=['Product', 'Region'])

def scale_features(df):
    scaler = StandardScaler()
    df['Sales'] = scaler.fit_transform(df[['Sales']])
    return df

if __name__ == "__main__":
    df = load_data('../data/sales_data.csv')
    df = handle_missing_values(df)
    df = remove_duplicates(df)
    df = handle_outliers(df)
    df = encode_categorical_variables(df)
    df = scale_features(df)
    df.to_csv('../data/cleaned_sales_data.csv', index=False)
    print("Data cleaning and preprocessing complete.")
