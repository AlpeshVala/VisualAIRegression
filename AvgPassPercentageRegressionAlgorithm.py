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

def applyDecisionTreeRegressionAlgorithm(metrics_test_data,X_train,X_test,y_train,y_test):
    decisionTreeRegModel = DecisionTreeRegressor(max_depth = 4)
    decisionTreeRegModel.fit(X_train,y_train)
    print('Score of the model is : ',decisionTreeRegModel.score(X_train,y_train))
    y_pred_dc_test = decisionTreeRegModel.predict(X_test)
    print('Predicted values of Average Pass Percentage after apply DC Regression model are :',y_pred_dc_test)
    #Calculating Sumof Square errors
    SSE_DCModel = np.sum((y_pred_dc_test - y_test)**2)
    print('Sum of Square errors is : ',SSE_DCModel)
    SST_DCModel = np.sum((y_test - np.men(y_train))**2)
    print('Total Sum of Square errors is: ',SST_DCModel)
    R2_DCModel = (1 - (SSE_DCModel /SST_DCModel))
    print('R2 Score is ',R2_DCModel)
    RMSE_DCModel = np.sqrt(mean_squared_error(y_test,y_pred_dc_test))
    print('RMSE value is : ',RMSE_DCModel)

    #Pruning the model
    parameters = {'max_depth' : [1,2,3,4,5,6]}
    dcGridModel = GridSearchCV(decisionTreeRegModel,parameters,cv=5,scoring='r2')
    dcGridModel.fit(X_train,y_train)
    print('Best Parameters are: ',dcGridModel.best_params_)
    print('Best score of Pruned model is: ',dcGridModel.best_score_)

    #AS max depth generated after pruning the data is 3, now building the model with depth 3
    dcPrunedModel = DecisionTreeRegressor(max_depth=3)
    dcPrunedModel.fit(X_train,y_train)
    print('Score of the DC Pruned model is: ',dcPrunedModel.score(X_train,y_train))
    y_pred_dc_pruned = dcPrunedModel.predict(X_test)
    print('Predicted Avg Pass Percentage values after applying pruned model is: ',y_pred_dc_pruned)

    #Calculating performance parameters of pruned model
    SSE_PrunedModel = np.sum((y_pred_dc_pruned - y_test) ** 2)
    print('Sum of Square errors after applying pruned model is : ',SSE_PrunedModel)
    SST_PrunedModel = np.sum((y_test - np.mean(y_train)) ** 2)
    print('Total Sum of Square errors is: ',SST_PrunedModel)
    R2_PrunedModel = (1 - (SSE_PrunedModel /SST_PrunedModel))
    print('R2 score of the pruned model is : ',R2_PrunedModel)
    RMSE_PrunedModel = np.sqrt(mean_squared_error(y_test,y_pred_dc_pruned))
    print('RMSE of the pruned model is : ',RMSE_PrunedModel)

    #Exporting the values to a csv file
    prediction = pd.DataFrame(data = {"TestCaseId":metrics_test_data['JiraId'],"TestCaseName":metrics_test_data['TestCaseName'],"TestCaseDescription":metrics_test_data['TestCaseDescription'],"Journey":metrics_test_data['Journey'],"ServiceComponent":metrics_test_data['MicroserviceComponent'],
                                      "PredictedAvgPassPercent":y_pred_dc_pruned})
    prediction.to_csv('prediction.csv',index = False)


test_metrics_data = pd.read_csv("Mention path of train data sheet")
updatedTestMetrics = loadDataAndAnalyzeTrainData(test_metrics_data)
X_train = updatedTestMetrics.copy()
X_train = X_train.drop(['AvgPassPercentage'],axis=1)
y_train = updatedTestMetrics['AvgPassPercentage']
print('Shape of X_train data is: ',X_train.shape)
print('Shape of y_train data is: ',y_train.shape)

metrics_test_data = pd.read_csv("Path of Test Data sheet")
metricsTestData = loadDataAndAnalyzeTestData(metrics_test_data)
X_test = metricsTestData.copy()
X_test = X_test.drop(['AvgPassPercentage'],axis = 1)
y_test = metricsTestData['AvgPassPercentage']
print('Shape of X_test data is : ',X_test.shape)
print('Shape of y_test data is : ',y_test.shape)

applyDecisionTreeRegressionAlgorithm(metrics_test_data,X_train,X_test,y_train,y_test)




