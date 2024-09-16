# Import libraries
from fastapi import FastAPI
from pydantic import BaseModel
import predict


class Comment(BaseModel):
    comment_body: str


app = FastAPI()

@app.get('/')
def index():
    return {'message': 'Sentiment Analysis ML API'}


@app.post('/sentiment/predict')
def predict_sentiment(data: Comment):
    data = data.comment_body

    pred = predict.predict_sentiment(data)

    return {
        'prediction': pred[0]
    }