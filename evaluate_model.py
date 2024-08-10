import pickle
import torch
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from model_training import SentimentDataset, SentimentModel
from data_preprocessing import load_data, preprocess_data, split_data

def evaluate_model(model, vectorizer, X_test, y_test):
    X_test_tfidf = vectorizer.transform(X_test).toarray()
    test_dataset = SentimentDataset(X_test_tfidf, y_test)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    all_preds = []
    model.eval()
    with torch.no_grad():
        for texts, labels in test_loader:
            texts = torch.tensor(texts, dtype=torch.float32)
            outputs = model(texts)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.tolist())

    accuracy = accuracy_score(y_test, all_preds)
    return accuracy

if __name__ == "__main__":
    # Load and preprocess data
    df = load_data()
    df = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(df)

    # Load the saved model
    input_dim = 5000  # Ensure this matches what was used during training
    hidden_dims = [512]  # Ensure this matches what was used during training
    best_model = SentimentModel(input_dim=input_dim, hidden_dims=hidden_dims)
    best_model.load_state_dict(torch.load("best_sentiment_model.pth"))

    # Load the saved vectorizer
    with open("best_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    # Evaluate the model
    accuracy = evaluate_model(best_model, vectorizer, X_test, y_test)
    print(f"Test Accuracy: {accuracy}")
