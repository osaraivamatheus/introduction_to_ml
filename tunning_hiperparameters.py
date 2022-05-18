## This python code is set to be main algorithm to 
## train and hiperparametrize a list of models

### Importing libraries
import pandas as pd
import numpy as np

### Models list
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import BernoulliNB
# from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

### Data pre processing
from modelling.pre_process import pre_processing
from sklearn.model_selection import train_test_split

### Hiperparametrization throught cross validation
from sklearn.model_selection import GridSearchCV 

### Compare models
from sklearn.metrics import fbeta_score, make_scorer

### Save trained models
from joblib import dump, load


###Timing
from datetime import datetime

svd = pd.read_csv('train.csv')

# Hiperparametrização de modelos
models = {
          'Logistic': LogisticRegression(),
          'Naive_Bayes': BernoulliNB(),
          'Random_Forest': RandomForestClassifier(),
          'XGBoost': XGBClassifier(),
          'MLP': MLPClassifier(),
          'SVC': SVC()          
}

### Setting list of parameters for each model

weights1 = {'0': .97, '1': .03}
weights2 = {'0': .5, '1': .5} #importance weights for some models

## Logistic model list of parameters
params_logit = {'penalty':['l1', 'l2', 'elasticnet'],
                'C':np.arange(0, 10, 1),
                'fit_intercept':[True],
                'class_weight':['balanced', None, weights1, weights2]
               }

### Support Vector Classifier list of parameters
params_svc = {
              'C':np.arange(1,15, .5), 
              'kernel':['linear', 'sigmoid', 'rbf'],
              'gamma':['scale', 'auto'],
              'class_weight':[weights1, weights2, 'balanced', None],
              'verbose':[True]              
             }

### Multi Layer Perceptron list of parameters
params_mlp = {
              'hidden_layer_sizes':[(10, 20), (50,40), (100, 80), (150, 100), (200, 150), (400, 200)],
              'activation':['logistic', 'tanh', 'relu'],
              'alpha': np.arange(.0001, .01, .001),
              'learning_rate':['constant', 'invscaling', 'adaptive'],
              'verbose':[True]
              }

### XGBoost list of parameters
params_xgb = {
              'n_estimators':[10, 50, 100, 150, 200, 400],
              'booster':['gbtree', 'gblinear', 'dart'],
              'importance_type':["gain", "weight", "cover", "total_gain","total_cover"],
              'eta':[.3, .5, .6],
              'gamma':[0, .2, .3],
              'max_depth':[50, 100, 150, 200, 500],
              'tree_method':['approx', 'hist, gpu_hist']
              }

### Random Forest list of parameters
params_rf = {
             'n_estimators': list(np.arange(10, 400, 50)),
             'criterion':["gini", "entropy"],
             'class_weight':["balanced", "balanced_subsample"],
             'max_features':['auto', 'sqrt', 'log2']
            }

## Nave Bayes list of parameters
params_nb = {
             'alpha':[1,2,10],
             'fit_prior':[True, False]    
            }


## Dictionary of all dictionary parameters
h_params = {
            'Logistic': params_logit,
            'Naive_Bayes': params_nb,
            'Random_Forest': params_rf,
            'XGBoost': params_xgb, 
            'MLP':params_mlp, 
            'SVC':params_svc
            }


### Setting all up
setups = [] #empty list to save GridSearch objects
f2 = make_scorer(fbeta_score , beta=2) #metric evaluation

### Setting up each model
for name in models.keys():
    model = GridSearchCV(estimator=models[name], 
                         param_grid=h_params[name], 
                         n_jobs=8, 
                         scoring=f2, verbose=2)
    setups.append(model)


# Pre processing data

df = pre_processing(svd)

df.select_columns(drop=['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked']) ## selecting features
df.Create_I_Matrix() # create incidence matrix
df.Standardize() # standardizing data
df.FactorCategorical() # labeling categorical featuers
df.RemoveOutliers() # removing outliers
df.fill_nan_ols('Age') # fill nan values with regression
train, test = train_test_split(df.df, 
                               test_size=.2, 
                               random_state=123, 
                               stratify=df.df['Survived']) #spliting data 

xtrain = train.loc[:, ~train.columns.isin(['Survived'])]
xtest = test.loc[:, ~test.columns.isin(['Survived'])]
ytrain = train['Survived']
ytest = test['Survived']


clf = [] # Empty list to save classifiers

### Running hiperparametrization
for name, modelo in zip(models.keys(), setups):
    print(f'Modelo {name}')
    t0 = datetime.now()
    m = modelo.fit(X=xtrain, y=ytrain)
    clf.append(m)
    dump(m, f'modelling/hiperparameters/{name}.joblib')
    print(f'\n Tempo de modelagem: {datetime.now() - t0}')

# dump(clf, 'All_models.joblib')
