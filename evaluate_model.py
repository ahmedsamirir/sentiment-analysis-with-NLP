import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from data_preprocessing import load_data, preprocess_data, split_data
from model_training import SentimentDataset, load_model

def evaluate_model(model, vectorizer, X_test, y_test):
    X_test_tfidf = vectorizer.transform(X_test).toarray()
    test_dataset = SentimentDataset(X_test_tfidf, y_test)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    all_preds = []
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient computation for evaluation
        for texts, _ in test_loader:
            texts = torch.tensor(texts, dtype=torch.float32)
            outputs = model(texts)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.tolist())

    accuracy = accuracy_score(y_test, all_preds)
    return accuracy

if __name__ == "__main__":
    # Load the saved model
    model_save_path = 'sentiment_model.pth'
    loaded_model = load_model(model_save_path=model_save_path)

    # Load and preprocess data (assuming the same preprocess and split functions from previous scripts)
    df = load_data()
    df = preprocess_data(df)
    _, X_test, _, y_test = split_data(df)

    # Evaluate the model
    vectorizer = TfidfVectorizer(max_features=5000)
    vectorizer.fit(df['text'])  # Fit the vectorizer on the full dataset text
    accuracy = evaluate_model(loaded_model, vectorizer, X_test, y_test)
    print(f"Model accuracy on test data: {accuracy:.2f}")
