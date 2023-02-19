# Sentiment Analysis - Restaurant Reviews - NLP Case study
This is a Natural Language Processing (NLP) case study for sentiment analysis of restaurant reviews. The objective of this project is to build a machine learning model that can predict whether a given restaurant review is positive or negative.

## Dataset
The dataset used for this project contains 1000 restaurant reviews, including their sentiment. The dataset is in the form of a TSV file where each row represents a review and has two columns, "Review" and "Liked." The "Review" column contains the text of the review, and the "Liked" column contains the sentiment, where 0 represents a negative sentiment, and 1 represents a positive sentiment.

## Libraries used
The following Python libraries are used in this project:

* numpy
* pandas
* matplotlib
* seaborn
* nltk
* scikit-learn

## Pre-processing data
Before building the model, we pre-processed the data by performing the following steps:

* Removing special characters and punctuation marks
* Converting all text to lowercase
* Removing stop words
* Lemmatizing the text

## Model building
We built two models for this project: Multinomial Naive Bayes and Linear Support Vector Classification (LinearSVC). The models were trained on the pre-processed data and evaluated using accuracy, confusion matrix, and classification report metrics.

## Model deployment
The LinearSVC model was chosen as the best performing model based on its accuracy score. The model was then saved using joblib and can be deployed and used for predicting sentiment on new restaurant reviews.

## Files included
The following files are included in this project:

* Jupyter Notebook (.ipynb) - Contains the code and comments for the project.
* Restaurant Reviews dataset (.tsv) - Contains the original dataset used for this project.
* LinearSVC model (.pkl) - Contains the trained LinearSVC model for sentiment analysis.

## Conclusion
This project demonstrates the use of NLP and machine learning techniques for sentiment analysis on restaurant reviews. The trained model can be used to predict the sentiment of new reviews and provide valuable insights for restaurant owners to improve their services.
