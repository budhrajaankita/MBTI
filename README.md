# MBTI

Introduction

The Myers-Briggs Type Indicator (MBTI) is a personality assessment tool used to determine personality types. The test categorizes people into 16 personality types based on four dichotomies: introversion/extroversion, intuition/sensing, thinking/feeling, and judging/perceiving.

Goal
The overall goal for the outcome is to predict a person’s personality type based on their social media posts and trying to address the following questions:
- Can Myers-Briggs personality type be predicted based on social media activity?
- Which Model performs the best and gives the most accurate results?
- What are the essential features of this data?
- Where can this model be used?

About the Dataset
Kaggle Dataset: This dataset contains over 8600 rows. Each row has a person’s personality Type (MBTI code) and a list of the last 50 things they’ve posted.

Data Preprocessing
1. Remove URLs, numbers, extra white spaces, special characters, etc
2. Apply lemmatizer.lemmatize() to return the lemma form of each word to get the tokenized text
3. Remove default English stopwords
4. Performed sentiment analysis (using TextBlob) of the clean tokenized text ( giving each a compound score for their 50 posts together)

Feature Engineering

- Word count
- Variance of word count
- Vocabulary richness ( no. of unique words / total number of words)
- Nouns, verbs, adjectives, interjections (using pos tag)
- Average links and images per post
- Average Question_marks, exclamations, ellipses per post
- 4 new categories for each pair - E/I, S/N, T/F, J/P
- Sentiment polarity score

Feature Importance using PCA and random forest

Models
Created a pipeline using imblearn package:
   Created a TfidfVectorizer for the tokenized text column.
   Used SelectKBest using the f_classif scoring function and the MinMaxScaler
   Used Under sampling because of class imbalance
   Added more stop words as part of preprocessing (as seen in WordCloud)

Types of Models
  Naive Bayes  
  Logistic Regression
    Lasso
    Ridge
    LogisticCV
  Random Forest

Evaluation Metrics used to compare models

ROC-AUC Score
Precision-Recall Score

Conclusion

- Heavily Imbalanced Data
- Didn’t work well for Extrovert-Introvert and Sensing-Intuition as most of them identified as Introvert and Intuitive.
- Regularization didn’t improve scores by a lot, but LogisticCV Regression worked best for this dataset.
In the future:
- also try Neural Networks which could improve the scores and can skip feature engineering by a lot.
- Could be used in improving marketing campaigns, understanding social media behavior, and styles of each type







