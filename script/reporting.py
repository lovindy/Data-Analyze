import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def load_data(filepath):
    return pd.read_csv(filepath)

def train_model(df):
    X = df.drop('Sales', axis=1)
    y = df['Sales']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')
    return model, X_test, y_test, y_pred

if __name__ == "__main__":
    df = load_data('../data/cleaned_sales_data.csv')
    model, X_test, y_test, y_pred = train_model(df)
    print("Model training and evaluation complete.")
