# Introduction to Sklearn models: a kaggle competition

This repository was created to introduce machine learning modeling with scikit learn for begginers. For this purpose, a Kaggle competition is used to explain what are the most classical steps in a classification modelling problem. 

## The problem

Basically, we need to build a model to predict which passengers survived the Titanic shipwreck. To do that, we will use some machine learning models that belongs to the supervised learning class of artificial intelligence. For more information about this competition and datasets, [check this link](https://www.kaggle.com/competitions/titanic).

## Exploratory data analisys

One of the most important steps on a modelling process is the Exploratory Data Analisys (EDA). Data distribution, data quality, new insights, variability analisys are made with EDA. This is importante because according to this analisys many other possibilities can arise, such as possibile models, inclusion of new variables, exclusion of unecessary variables and many other pre process tranformation. The EDA of this problem is found on EDA.ipynb file.

## Pre processing data

Pre processing data refers to all the transformations that must to be applied on the data before it is used on the modelling process. This is something specific to each situation and dataset available, but the EDA is an eficcient way to discover what transformations on the data must to be applied before it's passed to the modelling process. 

In the case of Titanic competition, the pre processing transformations applied on the data are:

1. Drop uncecessary features (columns);
2. Create a design matrix for the categorical variables (One hot encoding transformation);
3. Standartize numerical features;
4. Remove outliers;
5. Fill missing values.

Again, the tranformations used in the pre processing steps are specific to each problem of modelling and, obviously, it may change a lot.

For this problem it was created a python class that have methods to do all of those particular pre processing steps. This class is created on the script ["modelling/pre_process.py"](https://github.com/osaraivamatheus/titanic_with_Sklearn/blob/main/modelling/pre_process.py).

## Modelling

Since the EDA it's already made, it is possible to choose the models for data. In this exemple, the main task is to predict if a passenger survived or not, i.e., a binary classification problem. **A very useful tip: always start with the simpliest model.** Candidate models are:

- Logistic regression
- Naive Bayes classification
- Random Forest
- Multilayer Perceptron (NN)
- Extreme Gradiente Boosting (XGB)

### Hiperparametrization

For each possible model there are many parameters and hiperparameters to be estimated. The hiperparameters refers to the parameters that is related to the learning algorithm of each model. For exemple, in a neural network model, the number of hidden layers and nodes, learning hate, batch size are hiperparameters. 

One way to find the hiperparameters of a model is by the [Cross Validation technique](https://scikit-learn.org/stable/modules/cross_validation.html). In the Titanic problem, all the hiperparameters were estimated throught the [GridSerachCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) method in the Scikit Learn, setted up throught the script [tunning_hiperparameters.py](https://github.com/osaraivamatheus/titanic_with_Sklearn/blob/main/tunning_hiperparameters.py). This script save, for each model, all the best parameters found via cross validation on the exention .joblib. The hiperparametrized models are saved in the folder ["modelling/hiperparameters"](https://github.com/osaraivamatheus/titanic_with_Sklearn/tree/main/modelling/hiperparameters).





