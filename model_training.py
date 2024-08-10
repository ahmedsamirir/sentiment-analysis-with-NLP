import torch
import pickle
from torch.utils.data import DataLoader, Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from torch import nn, optim
from data_preprocessing import load_data, preprocess_data, split_data
from sklearn.model_selection import train_test_split

class SentimentDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = torch.tensor(texts, dtype=torch.float32)
        self.labels = torch.tensor(labels.to_numpy(), dtype=torch.long)  # Convert Series to NumPy array

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

class SentimentModel(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super(SentimentModel, self).__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.5))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 2))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

def train_model(X_train, y_train, X_val, y_val, vectorizer_params, model_params, training_params):
    # Vectorize the data
    vectorizer = TfidfVectorizer(**vectorizer_params)
    X_train_tfidf = vectorizer.fit_transform(X_train).toarray()
    X_val_tfidf = vectorizer.transform(X_val).toarray()

    # Create datasets and dataloaders
    train_dataset = SentimentDataset(X_train_tfidf, y_train)
    train_loader = DataLoader(train_dataset, batch_size=training_params['batch_size'], shuffle=True)

    val_dataset = SentimentDataset(X_val_tfidf, y_val)
    val_loader = DataLoader(val_dataset, batch_size=training_params['batch_size'], shuffle=False)

    # Initialize model, criterion, and optimizer
    model = SentimentModel(input_dim=model_params['input_dim'], hidden_dims=model_params['hidden_dims'])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=training_params['learning_rate'])

    # Training loop
    for epoch in range(training_params['epochs']):
        model.train()
        for texts, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Evaluate on validation set
    model.eval()
    val_preds = []
    with torch.no_grad():
        for texts, labels in val_loader:
            outputs = model(texts)
            _, preds = torch.max(outputs, 1)
            val_preds.extend(preds.tolist())

    accuracy = accuracy_score(y_val, val_preds)
    return model, vectorizer, accuracy

def find_best_model(X_train, y_train):
    best_accuracy = 0
    best_model = None
    best_vectorizer = None
    best_params = None

    vectorizer_params_options = [
        # {'max_features': 5000, 'ngram_range': (1, 1)},
        {'max_features': 5000, 'ngram_range': (1, 2)}
    ]
    model_params_options = [
        {'input_dim': 5000, 'hidden_dims': [512]}
        # {'input_dim': 5000, 'hidden_dims': [512, 128]}
    ]
    training_params_options = [
        # {'learning_rate': 0.001, 'batch_size': 32, 'epochs': 10},
        {'learning_rate': 0.0005, 'batch_size': 64, 'epochs': 30}
    ]

    for vectorizer_params in vectorizer_params_options:
        for model_params in model_params_options:
            for training_params in training_params_options:
                # Split data into training and validation sets
                X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
                model, vectorizer, accuracy = train_model(X_tr, y_tr, X_val, y_val, vectorizer_params, model_params, training_params)
                print(f"Params: {vectorizer_params}, {model_params}, {training_params} => Accuracy: {accuracy}")

                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model = model
                    best_vectorizer = vectorizer
                    best_params = (vectorizer_params, model_params, training_params)

    print(f"Best Params: {best_params} => Best Accuracy: {best_accuracy}")
    return best_model, best_vectorizer

if __name__ == "__main__":
    # Load and preprocess data
    df = load_data()
    df = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(df)

    # Find the best model
    best_model, best_vectorizer = find_best_model(X_train, y_train)

    # Assume best_vectorizer is your trained vectorizer
    with open("best_vectorizer.pkl", "wb") as f:
        pickle.dump(best_vectorizer, f)

    # Save the best model
    torch.save(best_model.state_dict(), "best_sentiment_model.pth")
    print("Best model saved.")
