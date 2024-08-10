import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from torch import nn, optim

from data_preprocessing import load_data, preprocess_data, split_data

class SentimentDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts  # Already a NumPy array
        self.labels = labels.values if isinstance(labels, pd.Series) else labels  # Convert labels to NumPy array if it's a Pandas Series

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = torch.tensor(self.texts[idx], dtype=torch.float32)  # Convert text to PyTorch tensor
        label = torch.tensor(self.labels[idx], dtype=torch.long)  # Convert label to PyTorch tensor with long dtype
        return text, label

class SentimentModel(nn.Module):
    def __init__(self, input_dim):
        super(SentimentModel, self).__init__()
        self.fc = nn.Linear(input_dim, 2)

    def forward(self, x):
        return self.fc(x)

def train_model(X_train, y_train, model_save_path='sentiment_model.pth'):
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train).toarray()

    train_dataset = SentimentDataset(X_train_tfidf, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    model = SentimentModel(input_dim=5000)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10):
        for texts, labels in train_loader:
            texts = torch.tensor(texts, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.long)

            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Save the model's state dictionary after training
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    return model, vectorizer

if __name__ == "__main__":
    # Load and preprocess data
    df = load_data()
    df = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(df)

    # Train the model
    model, vectorizer = train_model(X_train, y_train)
    print("Model training completed.")
