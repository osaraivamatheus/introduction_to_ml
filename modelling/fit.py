# Essentials
import pandas as pd
import numpy as np

import os
from os import listdir
from os.path import isfile, join

# Models list
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import BernoulliNB
from xgboost import XGBClassifier

# Predicton performance
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, f1_score, fbeta_score, recall_score

# Pre processing
from .pre_process import pre_processing

# For plotting
import matplotlib.pyplot as plt

# For graphs
import networkx as nx

# Timing
import datetime

# Save and load
# from pickle import dump, load
from joblib import dump, load


class ml_fitting:
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.trained = {}
        self.chosen_models = []
        self.hiperparameters = {}
        self.comparison = pd.DataFrame()

    def train_models(self, chosen_models=["Logistic", "Naive_Bayes"]):

        """
        This function train and saves models from
        a list of models already
        hiperparametrized.

        Parameters
        -----------

        X_train : pd.DataFrame.
                  DataFrame without targed column.

        y_train : pd.DataFrame.
                  DataFrame with only the target column.

        chosen_models: list
                        List of models to train. Avaiable values are:
                        'Logistic', 'Naive_Bayes', 'Random_Forest', 'SVC', 'MLP'
                        and 'XGBoost'.

        Returns
        -------
        The chosen models will be trained and saved
        in folder location specified with the .joblib
        extension.

        """

        # Load hiperparametrized models
        path = "./modelling/hiperparameters/"
        models = [f for f in listdir(path) if isfile(join(path, f))]
        avaiable_models = [m.split(".")[0] for m in models]

        print(avaiable_models)

        for i in chosen_models:
            if i not in avaiable_models:
                raise Exception(f"{i} model was not hiperparametrized yet.")

        print("All chosen models are available.")

        hiper_models = {}

        for name, model in zip(avaiable_models, models):
            hiper_models[name] = load(f"{path}{model}")

        print("The hiperparameters of each chosen model was loaded successfully.")

        pure_models = {
            "Logistic": LogisticRegression(),
            "Naive_Bayes": BernoulliNB(),
            "Random_Forest": RandomForestClassifier(),
            "XGBoost": XGBClassifier(),
            "MLP": MLPClassifier(),
            "SVC": SVC(),
        }

        to_train = {}
        h_params = {}

        # Subselection of chosen models
        for key, value in hiper_models.items():
            if key in chosen_models:
                h_params[key] = value  # Hiperparameters

        for key, value in pure_models.items():
            if key in chosen_models:
                to_train[key] = value  # chosen models

        # Training models
        print("Training models... \n")
        self.chosen_models = chosen_models
        best_params = {}
        for m in to_train.keys():
            model = pure_models[m].set_params(**h_params[m].best_params_)
            print(f"Hiperparameters of {m} model are settled. Fitting...")
            if m == "XGBoost":
                model.fit(self.X_train, self.y_train, eval_metric="logloss")
            else:
                model.fit(self.X_train, self.y_train)
            self.trained[m] = model
            dump(model, f"modelling/trained_models/{m}.joblib")  # saving trained model
            print(f"{m}: trained and saved. \n")
            best_params[m] = model.get_params()

        self.hiperparameters = best_params

    def compare_models(self, X_test, y_test, chosen_models=None):
        """
        Lightweight script to test and compare many models
        according to some metrics.

        Parameters
        ----------

        X_test: pd.DataFrame.
                DataFrame contaning the features to be classified.

        y_test: pd.DataFrame.
                DataFrame with only the target column.

        chosen_models: list. If None the models chosen in the train_models
                        method will be used.
                        List of models to compare. Avaiable values are:
                        'Logistic', 'Naive_Bayes', 'RF', 'SVC','MLP' and 'XGB'.
        Return
        ------

        DataFrame with evaluation metrics.
        """

        # Load available hiperparametrizated trained models
        path = "./modelling/trained_models/"
        models = [f for f in listdir(path) if isfile(join(path, f))]
        avaiable_models = [m.split(".")[0] for m in models]

        if chosen_models == None:
            chosen_models = list(self.trained.values())

        for i in chosen_models:
            if i not in avaiable_models:
                raise Exception("{} model was not trained yet.".format(i))

        # Subselecting chosen models
        trained_h_models = {}  # empty dict

        for name, model in zip(avaiable_models, models):
            trained_h_models[name] = load(f"{path}{model}")  # load all available

        selected_trained_h_models = {}

        for key, value in trained_h_models.items():
            if key in chosen_models:
                selected_trained_h_models[key] = value  # subselection all available

        # Empty list for storing metrics
        AUC = []
        RECALL = []
        fhalf = []
        f1 = []
        f2 = []
        model_name = []
        tp = []
        tn = []
        fp = []
        fn = []

        # Iterate over hiperparametrized trained models
        for model in selected_trained_h_models.keys():

            t0 = datetime.datetime.now()  # to calculate fitting time
            clf = selected_trained_h_models[model]  # get classifier
            y_pred = clf.predict(X_test)  # prediction
            cm = confusion_matrix(y_test, y_pred)  # confusion matrix
            model_name.append(model)  # appending model name

            # check for inconsistances
            if y_test.sum() == 0:
                AUC.append(np.nan)
                RECALL.append(np.nan)
                f1.append(np.nan)
                fhalf.append(np.nan)
                f2.append(np.nan)
                tp.append(np.nan)
                tn.append(np.nan)
                fp.append(np.nan)
                fn.append(np.nan)
            else:
                AUC.append(roc_auc_score(y_test, y_pred, average="weighted"))
                RECALL.append(recall_score(y_test, y_pred))
                fhalf.append(fbeta_score(y_test, y_pred, beta=0.5))
                f1.append(fbeta_score(y_test, y_pred, beta=1))
                f2.append(fbeta_score(y_test, y_pred, beta=2))
                tp.append(cm[1, 1])
                tn.append(cm[0, 0])
                fp.append(cm[0, 1])
                fn.append(cm[1, 0])

            t1 = datetime.datetime.now()
            print(f"Classificador: {model}, tempo de predição:{t1-t0} \n")

        # Creating DataFrame
        df = pd.DataFrame(
            {
                "MODEL": model_name,
                "AUC": AUC,
                "RECALL": RECALL,
                "FHALF": fhalf,
                "F1": f1,
                "F2": f2,
                "TN": tn,
                "FN": fn,
                "FP": fp,
                "TP": tp,
            }
        )

        df["N_NEG"] = sum(y_test == 0)
        df["N_POS"] = sum(y_test == 1)

        self.comparison = df

class predict_from_load_model:
    def __init__(self, X_test, choosen_models):
        self.X_test = X_test
        self.choosen_models = choosen_models
        self.hiperparameters = {}
        self.ids = X_test.index

    def do_it(self):
        """
        This functions make predictions of a data set given some loaded trained model. 
        
        Parameters:
        -----------
        X_test: pd.DataFrame.
                DataFrame contaning the features to be classified.
        
        choosen_models: list
                        List of models that will make the predictions. If None all
                        the available models will be used. The available values are:
                        'Logistic', 'Naive_Bayes','Random_Forest', 'MLP', 'XGBoost'
                        and 'SVC'.  
                        
        Return:
        -------
        DataFrame.
        
        """
        
        predictions = {}

        # Load available hiperparametrizated trained models
        path = "./modelling/trained_models/"
        models = [f for f in listdir(path) if isfile(join(path, f))]
        avaiable_models = [m.split(".")[0] for m in models]

        #         avaiable_models = [m for m in models if m in self.choosen_models]

        # Checking consistancy of models choosen
        if self.choosen_models == None:
            raise Exception(
                f"Select at least one trained model to make predicitons. The available models are: \n {avaiable_models}"
            )

        for trained in self.choosen_models:
            if trained not in avaiable_models:
                raise Exception(f"{trained} model was not trained yet.")

        # Loading choosen and trained models

        models_name = [m for m in avaiable_models if m in self.choosen_models]

        h_trained_models = {}  # empty dict

        for name, model in zip(models_name, models):
            h_trained_models[name] = load(f"{path}{model}")  # load all available

        for key, value in h_trained_models.items():
            if key in self.choosen_models:
                self.hiperparameters[key] = value  # subselection all available

        # Making predictions

        for model in h_trained_models.keys():
            predictions[model] = h_trained_models[model].predict(self.X_test)

        y = pd.DataFrame(data=predictions)
        y["PassengerId"] = self.ids

        return y
