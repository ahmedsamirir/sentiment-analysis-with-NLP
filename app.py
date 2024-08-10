from fastapi import FastAPI, HTTPException
import uvicorn
import torch
import pickle
import logging
from pydantic import BaseModel
from model_training import SentimentModel  # Make sure the path to SentimentModel is correct

# Set up logging
logging.basicConfig(filename="app.log",
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    level=logging.INFO)

app = FastAPI()

# Load the trained vectorizer and model
try:
    with open("best_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    # Define the model parameters (ensure these match what you used during training)
    input_dim = vectorizer.max_features
    hidden_dims = [512]  # Update this list based on your model's architecture

    # Initialize the model with the correct parameters
    model = SentimentModel(input_dim=input_dim, hidden_dims=hidden_dims)
    model.load_state_dict(torch.load("best_sentiment_model.pth"))
    model.eval()
    logging.info("Model and vectorizer loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model or vectorizer: {e}")
    raise RuntimeError(f"Error loading model or vectorizer: {e}")

# Define the request format
class SentimentRequest(BaseModel):
    text: str

# Define the prediction endpoint
@app.post('/predict')
def predict_sentiment(request: SentimentRequest):
    try:
        # Preprocess the input text
        text = request.text
        text_tfidf = vectorizer.transform([text]).toarray()
        text_tfidf = torch.tensor(text_tfidf, dtype=torch.float32)
        
        # Get the model prediction
        outputs = model(text_tfidf)
        _, preds = torch.max(outputs, 1)
        sentiment = 'positive' if preds.item() == 1 else 'negative'
        logging.info(f"Prediction result: {sentiment}")
        return {'sentiment': sentiment}
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

if __name__ == "__main__":
    logging.info("Starting FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
