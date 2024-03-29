# ReliePH Disaster Headline Classifier

The ReliePH Disaster Headline Classifier webscrape and classifies recent disaster news article headlines into various categories of actionable disaster events. It utilizes machine learning techniques and advanced text models to enable prompt and targeted relief action responses.


## Installation

Before running the program, make sure you have Python and pip installed on your system. Then, install the necessary Python modules using the following commands:

```bash
pip install fastapi pydantic joblib numpy pandas scikit-learn nltk
```
## Dataset

This project utilizes a dataset containing news article headlines on various types of disasters such as conflict, earthquake, fire, typhoon, and volcanic activities.

## Usage

To start the FastAPI server, run:

```bash
python3 -m uvicorn main:app
```


## Endpoints

### POST /classify

This endpoint accepts a JSON payload containing text data to classify. The input JSON should have the following structure:

```json
{
  "headline": "Text to classify"
}
```

The endpoint returns the predicted class label along with the probability distribution across different classes.

```json
{
  "prediction": "category"
}
```

## Model Training

The model is trained using a Multinomial Naive Bayes classifier with text preprocessing. The training script can be found in the `model.py` file.


## Preprocessing

Text preprocessing involves tasks such as tokenization, stop word removal, and lemmatization. This is handled using the preprocess_text function from the `preprocessing.py` file.

## Creadits

This project was developed by:

- **[Christian Morelos](https://github.com/ChanMor)**