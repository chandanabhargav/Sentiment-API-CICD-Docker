from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.model import sentiment_pipeline

app = FastAPI(title="Real-Time Sentiment Classifier")

class TextInput(BaseModel):
    text: str

@app.post("/predict")
def predict_sentiment(input: TextInput):
    try:
        result = sentiment_pipeline(input.text)[0]
        return {
            "label": result["label"],
            "score": round(result["score"], 4)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
