from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime
from typing import List

from headline import data
from classify import classify

app = FastAPI()

class HeadlineInput(BaseModel):
    headline: str

class PredictionOutput(BaseModel):
    prediction: str

class Headline(BaseModel):
    title: str
    link: str
    datetime: datetime
    disasterType: str

@app.get("/headlines", response_model=List[Headline])
async def scrape_headline_data():
    return data()

@app.post("/classify")
async def classify_headline(data: HeadlineInput):
    return classify(data.headline)
