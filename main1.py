import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix
import pandas as pd

# Load your CSV data
data = pd.read_csv("dataset.csv")

# Drop rows with missing values
data.dropna(inplace=True)

# Assuming your CSV has 'headline' and 'category' columns
headlines = data['headline']
categories = data['category']

print(categories.unique())

# Split your data into train and test sets
from sklearn.model_selection import train_test_split
train_headlines, test_headlines, train_categories, test_categories = train_test_split(headlines, categories, test_size=0.2, random_state=42)

# Create and train your model
model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(train_headlines, train_categories)

# Make predictions
labels = model.predict(test_headlines)

# Function to predict category
def predict_category(s, model=model):
    pred = model.predict([s])
    return pred[0]

# Example usage
print(predict_category("drawing paint"))

# Evaluate your model
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(test_categories, labels)
print("Accuracy:", accuracy)
