#Essentials
import pandas as pd
import numpy as np

#Data pre processing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression, LinearRegression

#Oversampling
from imblearn.over_sampling import SMOTENC

class pre_processing():
    """
    This is a class that provides chaining functions
    to excecute the modelling data pre processing. This
    initial processing following the steps:
    
    1) RemoveOutliers(): Remove outliers from chosen
    variables.
    2) Standardize(): Apply a transformation on 
    numerical variables, such standartization.
    3) FactorCategorical(): Transform all the categorical.
    variables into factor variables (labelling).
    4) Split(): Split data into train and test sets.
    
    """
    
    def __init__(self, df): 
        self.df = df.copy()
        self.selected_columns = self.df.columns.tolist()
        
    def select_columns(self, drop, return_df=False):
        """
        This funcion was created to select only the features used on modelling process.
        
        Parameters:
        -----------
        drop : columns to drop.
        
        return_df : bool, default False.
                    If True a data frame without
                    outliers is returned.
        
        
        Returns
        -------
        DataFrame
            DataFrame or method.
        
        """
        
        self.df.drop(columns=drop, axis=1, inplace=True)
        return self
        
    def RemoveOutliers(self, outlier_columns=['Fare'], return_df=False):
        """
        This functions removes the outliers from the columns 
        passed in outlier_colunmn list. If an outlier is 
        detected, the entire row of data frame is deleted. 
        
        The outliers are identified throught de Isolation 
        Forest Algorithm (Liu, et al. 2008)
        
        Parameters:
        -----------
        outlier_columns : list of columns to 
                          search for outliers.
                          
        return_df : bool, default False.
                    If True a data frame without
                    outliers is returned.
                   
        Returns
        -------
        DataFrame
            DataFrame or method.
            
        References
        ----------
        
        Liu, Fei Tony, Ting, Kai Ming and Zhou, Zhi-Hua. 
        “Isolation forest.” Data Mining, 2008. ICDM’08. 
        Eighth IEEE International Conference on.
        -----------
        """
        # Detecting and removing outliers of transactions 
        # values throught IForest
            
        print('Detecting and removing outliers...')
        if len(outlier_columns) == 1:
            outs = np.array(self.df[outlier_columns]).reshape(-1,1)
        else:
            outs = np.array(self.df[outlier_columns])
        iso = IsolationForest(random_state=123)
        outs = iso.fit_predict(outs)
        self.df = self.df[outs == 1].copy()

        print(f'Total rows deleted containing outliers: {sum(outs == -1)}')

        if return_df:
            return self.df

        return self
    
    def Standardize(self, return_df=False):
        """
        This function provides a transformation on numerical data.
        The trasformation given by z, such that
        
        z = (x - mu)/s ,
        
        where x is the observation, mu is the mean and s is the stardart deviation.
        
        Parameters
        ----------
        return_df : bool, default False.
                    If True a data frame with
                    numerical features standadized
                    is returned.
                   
        Returns
        -------
        DataFrame
            DataFrame or method.
        
        """
        
        # Numerical variables except the depend one
        numericals = ['int16', 'int32', 'float16', 'float32', 'float64']
        
        # Standarting
        std = StandardScaler()
        numerical_variables_sdt = std.fit(self.df.select_dtypes(include=numericals))
        numerical_variables_sdt = std.transform(self.df.select_dtypes(include=numericals))
        
        numeric_columns = self.df.select_dtypes(include=numericals).columns
        self.df[numeric_columns] = numerical_variables_sdt
        
        if return_df:
            return self.df
        
        return self
    

    def FactorCategorical(self, return_df=False):
        """
        This function factorise the categorical variables on data frame. 
        
        Parameters
        ----------
        return_df : bool, default False.
                    If True a data frame with
                    categorical features labeled
                    is returned.
                   
        Returns
        -------
        DataFrame
            DataFrame or method.
        
        """
                
       
        # Categorical variables except the depend one
        categoricals = self.df.select_dtypes(include=['category', 'O']).columns
                
        
        # Factorizing categorical data:
        for i in categoricals:
            enc = LabelEncoder()
            enc.fit(self.df[i].apply(str))
            self.df[i] = enc.transform(self.df[i].apply(str))
            
        
        if return_df:
            return self.df
        
        return self
    
    def Create_I_Matrix(self, target='Survived', return_df=False):
        """
        This function generates an incidence matrix of the 
        categoricals variables and put them into original data frame.
        
        Parameters
        ----------
        return_df : bool, default False.
                    If True a data frame with
                    categorical features labeled
                    is returned.
                   
        Returns
        -------
        DataFrame
            DataFrame or method.
        
        """
                
       
        # Categorical variables except the depend one
        categoricals = self.df.select_dtypes(include=['category', 'O', 'int64']).columns
        categoricals = list(categoricals)
        if target in categoricals:
            categoricals.remove(target)
        
        self.df = pd.get_dummies(self.df, columns = categoricals)
        self.selected_columns = self.df.columns.tolist()
        
        if return_df:
            return self.df
        
        return self
    
    def fill_nan_ols(self, y):
       
        na = self.df[y].isna()
        train = self.df[~na]
        test = self.df[na]

        X_train_features = train.loc[:, ~train.columns.isin([y])]
        X_test_features = test.loc[:, ~test.columns.isin([y])]

               
        ols = LinearRegression().fit(X=X_train_features, y=train[y])
        y_hat = ols.predict(X_test_features)

        self.df.loc[na, y] = y_hat

        
        
        
        
        
            
        
        
      