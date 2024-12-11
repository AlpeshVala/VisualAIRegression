import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
from sklearn import preprocessing

def loadDataAndAnalyzeTrainData(test_metrics_data):
    updatedTestMetrics = test_metrics_data.copy()
    updatedTestMetrics = updatedTestMetrics.drop(['JiraId','TestCaseName','TestCaseDescription','TestCaseLabel','AutomationStatus'],axis = 1)
    print('Columns after deleting unwanted data are: ',updatedTestMetrics.columns)

    #Apply label encoding for categorical variables
    label_encoder = preprocessing.LabelEncoder()
    print('Unique release version values are: ',updatedTestMetrics['ReleaseVersion'].unique())
    updatedTestMetrics['ReleaseVersion'] = label_encoder.fit_transform(updatedTestMetrics['ReleaseVersion'])
    print('Unique values of Release Version after Label encoding are : ',updatedTestMetrics['ReleaseVersion'].unique())

    #Journey preprocessing
    updatedTestMetrics['Journey'] = label_encoder.fit_transform(updatedTestMetrics['Journey'])
    print('Unique values of Journey after Label encoding are : ', updatedTestMetrics['Journey'].unique())

    #Microservice Preprocessing
    updatedTestMetrics['MicroserviceComponent'] = label_encoder.fit_transform(updatedTestMetrics['MicroserviceComponent'])
    print('Unique values of MicroserviceComponent after Label encoding are : ', updatedTestMetrics['MicroserviceComponent'].unique())

    #Error Prone Preprocessing
    updatedTestMetrics['ErrorProne'] = label_encoder.fit_transform(updatedTestMetrics['ErrorProne'])
    print('Unique values of ErrorProne after Label encoding are : ', updatedTestMetrics['ErrorProne'].unique())

    # TestScriptPriority Preprocessing
    updatedTestMetrics['TestScriptPriority'] = label_encoder.fit_transform(updatedTestMetrics['TestScriptPriority'])
    print('Unique values of TestScriptPriority after Label encoding are : ', updatedTestMetrics['TestScriptPriority'].unique())

    #Sample Values
    print('Sample Values are : ', updatedTestMetrics.head(3))
    return updatedTestMetrics

def loadDataAndAnalyzeTestData(test_metrics_data):
    updatedTestMetrics = test_metrics_data.copy()
    updatedTestMetrics = updatedTestMetrics.drop(['JiraId','TestCaseName','TestCaseDescription','TestCaseLabel','AutomationStatus'],axis = 1)
    print('Columns after deleting unwanted data are: ',updatedTestMetrics.columns)

    #Apply label encoding for categorical variables
    label_encoder = preprocessing.LabelEncoder()
    print('Unique release version values are: ',updatedTestMetrics['ReleaseVersion'].unique())
    updatedTestMetrics['ReleaseVersion'] = label_encoder.fit_transform(updatedTestMetrics['ReleaseVersion'])
    print('Unique values of Release Version after Label encoding are : ',updatedTestMetrics['ReleaseVersion'].unique())

    #Journey preprocessing
    updatedTestMetrics['Journey'] = label_encoder.fit_transform(updatedTestMetrics['Journey'])
    print('Unique values of Journey after Label encoding are : ', updatedTestMetrics['Journey'].unique())

    #Microservice Preprocessing
    updatedTestMetrics['MicroserviceComponent'] = label_encoder.fit_transform(updatedTestMetrics['MicroserviceComponent'])
    print('Unique values of MicroserviceComponent after Label encoding are : ', updatedTestMetrics['MicroserviceComponent'].unique())

    #Error Prone Preprocessing
    updatedTestMetrics['ErrorProne'] = label_encoder.fit_transform(updatedTestMetrics['ErrorProne'])
    print('Unique values of ErrorProne after Label encoding are : ', updatedTestMetrics['ErrorProne'].unique())

    # TestScriptPriority Preprocessing
    updatedTestMetrics['TestScriptPriority'] = label_encoder.fit_transform(updatedTestMetrics['TestScriptPriority'])
    print('Unique values of TestScriptPriority after Label encoding are : ', updatedTestMetrics['TestScriptPriority'].unique())

    #Sample Values
    print('Sample Values are : ', updatedTestMetrics.head(3))
    return updatedTestMetrics

