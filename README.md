# Introduction to classification modelling: a kaggle competition

This repository was created to introduce machine learning modeling process begginers. For this purpose, a Kaggle competition is used to explain what are the most classical steps in a classification modelling problem. 

The python scripts and jupyter notebooks created uses many methods from libraries like [Pandas](https://pandas.pydata.org/), [Numpy](https://numpy.org/) as [Scikit Learn](https://scikit-learn.org/stable/).  

To all of this make sense to the reader, please follow these reading order:

1. This Readme file.
2. [EDA notebook](https://github.com/osaraivamatheus/titanic_with_Sklearn/blob/main/EDA.ipynb).
3. Run the [hiperparametrization script](https://github.com/osaraivamatheus/titanic_with_Sklearn/blob/main/tunning_hiperparameters.py).
4. [Comparison models notebook](https://github.com/osaraivamatheus/titanic_with_Sklearn/blob/main/Compare_models.ipynb).
5. Make predictions and export it this [notebook](https://github.com/osaraivamatheus/titanic_with_Sklearn/blob/main/Make_predictions.ipynb).


## The problem

Basically, we need to build a model to predict which passengers survived the Titanic shipwreck. To do that, we will use some machine learning models that belongs to the supervised learning class of artificial intelligence. For more information about this competition and datasets, [check this link](https://www.kaggle.com/competitions/titanic).

## Exploratory data analisys

One of the most important steps on a modelling process is the Exploratory Data Analisys (EDA). Data distribution, data quality, new insights, variability analisys are made with EDA. This is importante because according to this analisys many other possibilities can arise, such as possibile models, inclusion of new variables, exclusion of unecessary variables and many other pre process tranformation. The EDA of this problem is found on [EDA.ipynb](https://github.com/osaraivamatheus/titanic_with_Sklearn/blob/main/EDA.ipynb) file.

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

**Simple models**:
- Logistic regression
- Naive Bayes classification

**More complex models**
- Multilayer Perceptron (NN)
- Random Forest
- Extreme Gradient Boosting (XGB)

### Hiperparametrization

For each possible model there are many parameters and hiperparameters to be estimated. The hiperparameters refers to the parameters that is related to the learning algorithm of each model. For exemple, in a neural network model, the number of hidden layers and nodes, learning hate, batch size are hiperparameters. 

One way to find the hiperparameters of a model is by the [Cross Validation technique](https://scikit-learn.org/stable/modules/cross_validation.html). In the Titanic problem, all the hiperparameters were estimated throught the [GridSerachCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) method in the Scikit Learn, setted up throught the script [tunning_hiperparameters.py](https://github.com/osaraivamatheus/titanic_with_Sklearn/blob/main/tunning_hiperparameters.py). This script save, for each model, all the best parameters found via cross validation on the extention .joblib. The hiperparametrized models are saved in the folder ["modelling/hiperparameters"](https://github.com/osaraivamatheus/titanic_with_Sklearn/tree/main/modelling/hiperparameters).

A very very important thing to be connsidered about hiperparametrization process is the evaluation metrics. There are many of them and it is important to know what they are when to use each one of them.

### Training models

Alright! Now we have a set of hiperparameters for each choosen models. The next step is to train those models usings this hiperparameters to estimate all the "normal" parameters (ex.: the $\beta's$ in a logistic regression). This step is what defines a supervised model class, i.e, have a labelled data set available to train some models. There are at least other two types of machine learning models, such unsupervised and the reinforcement learning models. For more about that, I highly recomend to watch the [Alexander Amini classes from MIT](http://introtodeeplearning.com/).

To train our models using a set of hiperparameters previously estimated, we'll use the class ml_fitting from the script [fit.py](https://github.com/osaraivamatheus/titanic_with_Sklearn/blob/main/modelling/fit.py) in the folder "modelling/". On this python class there are some methods to train and evaluate a set of choosen models according to some pre defined metrics. A very good way to start thinking about evaluation metrics for binary classification is following the [flowchart](https://machinelearningmastery.com/tour-of-evaluation-metrics-for-imbalanced-classification/):

![How-to-Choose-a-Metric-for-Imbalanced-Classification-latest](https://user-images.githubusercontent.com/34166879/169283235-1a748519-1fec-4705-9eac-e8b0b1e9bfbc.png)


The training steps are explained on the noteboook [Compare_models.ipynb](https://github.com/osaraivamatheus/titanic_with_Sklearn/blob/main/Compare_models.ipynb). This notebook will train and save the choosen models. Also, it is shown all the available metrics to evaluate and compare the performance of each model. 

### Making predictions

At this moment, we've know the datasets, made EDA, choosen some classification models, hiperparametrized them, save them, train and compare all of them. Thus, we are now on the final step of a modelling process: the predictions, and we will use the best model to make it. For this purpose, we'll use the method *do_it* from the class *predict_from_load_model* to generate predictions. We could use a loaded model directly to make predicitons, but using this class brings some advantages, such as saving Id informations and show easily the hiperparameters. All of this activities are explained on the [Make_predictions.ipynb notebook](https://github.com/osaraivamatheus/titanic_with_Sklearn/blob/main/Make_predictions.ipynb). 




