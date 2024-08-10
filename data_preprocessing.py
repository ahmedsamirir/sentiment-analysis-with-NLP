import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_data():
    df = pd.read_csv('Data/sentiment_data.csv')
    return df

def preprocess_data(df):
    df['text'] = df['review'].str.lower()
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['sentiment'])
    return df

def split_data(df):
    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)
    return X_train.reset_index(drop=True), X_test.reset_index(drop=True), y_train.reset_index(drop=True), y_test.reset_index(drop=True)

