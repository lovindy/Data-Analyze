import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def load_data(filepath):
    return pd.read_csv(filepath)

def descriptive_statistics(df):
    print(df.describe())

def visualize_data(df):
    # Histogram of Sales
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Sales'], kde=True, bins=30)
    plt.title('Sales Distribution')
    plt.xlabel('Sales')
    plt.ylabel('Frequency')
    plt.show()

    # Box Plot of Sales
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df['Sales'])
    plt.title('Sales Distribution')
    plt.xlabel('Sales')
    plt.show()

if __name__ == "__main__":
    df = load_data('../data/cleaned_sales_data.csv')
    descriptive_statistics(df)
    visualize_data(df)
    print("EDA complete.")
