from typing import Dict

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI
from pydantic import BaseModel
from tokenizers import Tokenizer

from src.scripts.settings import Settings

SENTIMENT_MAP: Dict[int, str] = {
    0: "negative",
    1: "neutral",
    2: "positive",
}


settings = Settings()

tokenizer = Tokenizer.from_file(str(settings.onnx_tokenizer_path))

embedding_session = ort.InferenceSession(str(settings.onnx_embedding_model_path))

classifier_session = ort.InferenceSession(str(settings.onnx_classifier_path))


class TextIn(BaseModel):
    text: str


class PredictionOut(BaseModel):
    label: str


app = FastAPI()


@app.post("/predict", response_model=PredictionOut)
def predict(payload: TextIn) -> PredictionOut:
    cleaned_text = payload.text.strip()

    encoded = tokenizer.encode(cleaned_text)

    input_ids = np.array([encoded.ids], dtype=np.int64)
    attention_mask = np.array([encoded.attention_mask], dtype=np.int64)

    embedding_inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }
    embeddings = embedding_session.run(None, embedding_inputs)[0]

    classifier_input_name = classifier_session.get_inputs()[0].name
    classifier_inputs = {
        classifier_input_name: embeddings.astype(np.float32),
    }
    prediction = classifier_session.run(None, classifier_inputs)[0]

    pred_idx = int(prediction[0])
    label = SENTIMENT_MAP.get(pred_idx, "unknown")

    return PredictionOut(label=label)

