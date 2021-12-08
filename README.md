# RecSys 2019 Trivago Challenge

This is the code for my attempt at RecSys 2019 Challenge hosted by Trivago.

***

## Problem statement 
**trivago** is a global hotel search platform, where users can browse hotels based on their search criteria and click through to other booking sites to make a booking. The goal of the challenge is to predict a user's final clickout item given a list of hotel items and user's session activity.

## Solution summary
I cast the problem into a binary classification task by aggregating session logging data. Within a session, if an item was clicked out, it is marked as a positive example and all other items are marked as negative examples. The combination of extensive feature engineering and a gradient boosted decision trees model worked well for this task. Specifically, I used both LightGBM's binary classifier and learning-to-rank model. The features are focused on session interactions and item's global characteristics. The resulting model is one of user-independent and session-based nature. The final recommendations were formed by sorting the predicted probabilities that an item gets clicked out within a session. 

## Results and further improvements
Due to limited computing resources, I only used part of the logging data for training. The final evaluation was done based on an ensemble of LightGBM binary classifier and LightGBM ranker models. See the table below for the model's performance in mean reciprocal rank, which was the metric used in the challenge. Given more time and computing resources, I would have experimented using a large training set and perform more detailed hyperparameter tuning. More sophisticated features specific to recommendation systems can also be explored further.

| Model | Validation MRR | Confirmation MRR |
| ------------- | ------------- | ----- |
| Classifier | 0.672 | 0.669 |
| Ranker | 0.674 | 0.672 |
| Ensemble | 0.674 | 0.672 |

## Reproducing the solution
- Put competition datasets into data folder (train.csv/test.csv/validation.csv/confirmation.csv/item_metadata.csv)
- Run ```python 3 data_prep.py```
- Run ```python 3 lgb_cv.py```
- Data preparation takes about 2 hours
- 5-fold cross validation takes about 1 hour each for both the classifier and ranker model
- main library version
    - pandas==1.3.3
    - numpy==1.20.3
    - lightgbm==3.2.1
    - sklearn==1.0
