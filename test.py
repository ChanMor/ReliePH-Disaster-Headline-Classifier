import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# Assuming you have your own dataset with features (X) and labels (y)
# Replace X and y with your own dataset
X = [
    "This is a sample text about sports.",
    "I love hiking in the mountains.",
    "Politics can be a divisive topic.",
    "The latest technology trends are fascinating.",
    "I enjoy cooking new recipes.",
    "Books are a great source of knowledge.",
    "Music has the power to evoke emotions.",
    "Climate change is a pressing issue.",
    "I'm passionate about environmental conservation.",
    "Art is subjective and open to interpretation."
]

y = [
    "sports",
    "outdoor activities",
    "politics",
    "technology",
    "cooking",
    "education",
    "music",
    "environment",
    "environment",
    "arts"
]

# Split your dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline with TF-IDF vectorizer and Multinomial Naive Bayes classifier
model = make_pipeline(TfidfVectorizer(), MultinomialNB())

# Train the model on the training set
model.fit(X_train, y_train)

# Predict labels for the test set
labels = model.predict(X_test)

# Evaluate the model
mat = confusion_matrix(y_test, labels)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.show()

def predict_category(s, model=model):
    pred = model.predict([s])
    return pred[0]

print(predict_category("disaster"))
