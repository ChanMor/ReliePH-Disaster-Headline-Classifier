# ReliePH Disaster Headline Classifier

The ReliePH Disaster Headline Classifier webscrape and classifies recent disaster news article headlines into various categories of actionable disaster events. It utilizes machine learning techniques and advanced text models to enable prompt and targeted relief action responses.


## Installing External Modules

Before running the program, make sure you have Python and pip installed on your system. Then, install the necessary Python modules using the following commands:

```bash
pip install numpy matplotlib seaborn scikit-learn pandas
```


## Running The Program

To run the program, execute the provided Python script. Ensure you have a CSV file named dataset.csv containing the cleaned and preprocessed news article headlines with corresponding categories.

```bash
python main.py
```
## Overview

The program performs the following steps:

1. Data Preprocessing: Clean and preprocess the news article headlines using Pandas to ensure data integrity. It also uses the preprocessed headlines as seed data to generate synthetic text data to serve as a training dataset using the advanced text model - GPT-3.5 Turbo.

2. Model Training: Trains a machine learning model utilizing the Multinomial Naive Bayes classifier (MNB) to classify headlines into various categories of disaster events.

3. Model Evaluation: Evaluates the trained model's performance using accuracy metrics.

4. Prediction: Predicts the category of new headlines using the trained model.