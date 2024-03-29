import os
import re
import nltk
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    text = text.lower()

    tokens = nltk.word_tokenize(text)
    
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
    
    preprocessed_text = ' '.join(lemmatized_tokens)
    return preprocessed_text


def preprocess_csv_files(folder_path):
    for file in os.listdir(folder_path):
        if file.endswith('.csv'):

            df = pd.read_csv(os.path.join(folder_path, file))

            df['headline'] = df['headline'].apply(preprocess_text)
            
            df.to_csv(f'preprocessed_{file}', index=False)


preprocess_csv_files('dataset')

