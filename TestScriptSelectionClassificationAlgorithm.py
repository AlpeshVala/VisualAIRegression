import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score,auc,roc_curve,confusion_matrix,classification_report
from sklearn import tr+ee
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClasssifier

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

    #Sample Values
    print('Sample Values are : ', updatedTestMetrics.head(3))
    return updatedTestMetrics

def applyRandomForestClasssifierModel(X_train,X_test,y_train,y_test):
    rfModel = RandomForestClassifier(n_estimators=100,max_depth=4,max_features='auto',oob_score=True,verbose=1,random_state=50)
    rfModel.fit(X_train,y_train)
    print('Score of the model is ',rfModel.score(X_train,y_train))
    data = pd.series(data=rfModel.feature_importances_,index=X_train.columns)
    data.sort_values(ascending=True,inplace=True)
    y_pred_test_rf = rfModel.predict(X_test)
    print('Confusion Matrix is : ',confusion_matrix(y_test,y_pred_test_rf))
    print('Accuracy Score is : ',accuracy_score(y_test,y_pred_test_rf))

    #Perform Hyperparameter tuning
    parameters = {'max_features':[1,2,3,4,5,6,7,8,9],'max_depth':[1,2,3,4,5]}
    rfTunedModel = GridSearchCV(rfModel,parameters,cv=5,scoring='accuracy')
    rfTunedModel.fit(X_train,y_train)
    print('Best parameters are : ',rfTunedModel.best_params_)
    y_pred_test_rf_cv = rfTunedModel.predict(X_test)
    print('Predicted values after applying RF Pruned model are : ',y_pred_test_rf_cv)
    print('Accuracy score of the RF Tuned model is: ',accuracy_score(y_test,y_pred_test_rf_cv))

    rfClassificationReport = classification_report(y_test,y_pred_test_rf_cv)
    print('Classification report for RF model is : ',rfClassificationReport)

    regressionOutputData = pd.read_csv('path of prediction.csv file')
    regressionOutputData.insert('colno',column = "PredictedTest+S+criptPriority",value=y_pred_test_rf_cv)
    regressionOutputData.to_csv('mention path of FinalPrediction.csv',index=False)

test_metrics_data = pd.read_csv("Mention path of train data sheet")
updatedTestMetrics = loadDataAndAnalyzeTrainData(test_metrics_data)
X_train = updatedTestMetrics.copy()
X_train = X_train.drop(['++'],axis=1)
y_train = updatedTestMetrics['TestScriptPriority']
print('Shape of X_train data is: ',X_train.shape)
print('Shape of y_train data is: ',y_train.shape)

metrics_test_data = pd.read_csv("Path of Test Data sheet")
metricsTestData = loadDataAndAnalyzeTestData(metrics_test_data)
X_test = metricsTestData.copy()
X_test = X_test.drop(['TestScriptPriority'],axis = 1)
y_test = metricsTestData['TestScriptPriority']
print('Shape of X_test data is : ',X_test.shape)
print('Shape of y_test data is : ',y_test.shape)

applyRandomForestClasssifierModel(X_train,X_test,y_train,y_test)
