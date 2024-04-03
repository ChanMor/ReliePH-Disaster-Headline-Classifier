from joblib import load
from preprocessing import preprocess_text
import numpy as np

# Load the pre-trained model
model = load('disaster_classifier.joblib')

def classify(data):
    headline = data

    preprocessed_headline = preprocess_text(headline)

    prediction_probabilities = model.predict_proba([preprocessed_headline])[0]
    
    # Get the index of the predicted category with the highest probability
    max_prob_index = np.argmax(prediction_probabilities)
    
    # Get the corresponding predicted category
    predicted_category = model.classes_[max_prob_index]
    
    # Get the probability score of the predicted category
    prediction_score = prediction_probabilities[max_prob_index]
    
    # Set a minimum prediction score threshold
    min_prediction_score_threshold = 0.7
    
    if prediction_score >= min_prediction_score_threshold:
        return {"prediction": predicted_category}
    else:
        return {"prediction": "non-disaster"}