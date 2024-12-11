import pandas as pd
from rake_nltk import Rake
from nltk.corpus import stopwords

def filterKeywordsRakeMethod():
    nltk_words = list(stopwords.words('english'))
    new_stopwords = nltk_words

    #Read the user story file for content
    file1 = open("","r+")
    keywordText = file1.read()
    rakeObj = Rake(stopwords=new_stopwords,include_repeated_phrases=False)
    rakeObj.extract_keywords_from_text(keywordText)
    print('Keywordswith scroes using Rake algorithm for text are : ',rakeObj.get_ranked_phrases_with_scores())
    rakeKeywords = rakeObj.get_ranked_phrases_with_scores()
    rakeKeywordsDict = {}
    dictAccumlator = 1
    for d in rakeKeywords:
        for key in d:
            print(d[1])
            rakeKeywordsDict[dictAccumlator] = d[1]
            dictAccumlator += 1
    rakeKeywordsLst = [i for i in rakeKeywordsDict.values()]
    rakeKeywordsUniqueLst = list(dict.fromkeys(rakeKeywordsLst))
    print('Unique keyword values from List are : ',rakeKeywordsUniqueLst)

    #Removing keywords with length less than 1
    filteredRakeKeywordLst = []
    singleRakeKeywordLst = []
    for keyword in rakeKeywordsUniqueLst:
        if(len(keyword.split()) > 1):
            filteredRakeKeywordLst.append(keyword)
        else:
            singleRakeKeywordLst.append(keyword)

    #Logic to reverse the 2 wordssentence and append it to the filtered list
    reverseRakeKeywordLst = []
    for filteredKeyword in filteredRakeKeywordLst:
        if(len(filteredKeyword.split()) == 2):
            words = filteredKeyword.split(' ')
            reverseWords = ' '.join(reversed(words))
            reverseRakeKeywordLst.append(reverseWords)
    filteredRakeKeywordLst = filteredRakeKeywordLst + reverseRakeKeywordLst
    return filteredRakeKeywordLst


def validateKeywordsInTestScripts(testDescriptionDf,filteredRakeKeywordLst):
    testDescLst = testDescriptionDf.values.tolist()
    formattedTestDesc = []
    scriptsIncludedInRegression = []
    flag = 2
    for testDesc in testDescLst:
        print('Test Script Description value is: ',testDesc)
        if '_' in testDesc:
            testDesc = testDesc.replace("_"," ")
        if '-' in testDesc:
            testDesc = testDesc.replace("-"," ")
        formattedTestDesc.append(testDesc)
    testDescriptionDf = pd.DataFrame(data = {"UniqueTestDescriptionValues": formattedTestDesc})
    testDescriptionDf.to_csv("path to file")
    for testDescription in formattedTestDesc:
        for keywordPhrase in filteredRakeKeywordLst:
            if keywordPhrase.lower() in testDescription.lower():
                flag = 1
                scriptsIncludedInRegression.append(testDescription)
                break
            else:
                flag= 2
    regressionDecisionLst = []
    flag1 = 2
    for originalTestScript in testDescriptionDf['UniqueTestDescriptionValue']:
        for regressionScript in scriptsIncludedInRegression:
            if regressionScript.lower() == originalTestScript.lower():
                flag1 = 1
                regressionDecisionLst.append('Keyword found in Test Script Description')
                break
            else:
                flag1 = 2
        if(flag1 == 2):
            regressionDecisionLst.append('Keyword not found in Test Script Description')
    testDescriptionDf.insert(1,column="RegressionInclusionDecision",value=regressionDecisionLst)
    testDescriptionDf.to_csv("File Path")


def validateKeywordsInTestScriptName(testNameDf,filteredRakeKeywordLst):
    testNameDfLst = testNameDf.values.tolist()
    formattedTestNameDf = []
    scriptsIncludedInRegressionNme = []
    flag = 2
    for testScriptName in testNameDfLst:
        testScriptName = testScriptName.replace("_"," ")
        testScriptName = testScriptName.replace("-"," ")
        formattedTestNameDf.append(testScriptName)
    testNameDf = pd.DataFrame(data = {"UniqueTestNameValues":formattedTestNameDf})
    testNameDf.to_csv("Specify file path")
    #Checkfor presence of keywords in test script name

    for testCaseName in formattedTestNameDf:
        for keywordPhrase in filteredRakeKeywordLst:
            if keywordPhrase.lower() in testCaseName.lower():
                flag = 1
                print('Test script should be included in regression based on name')
                scriptsIncludedInRegressionNme.append(testCaseName)
                break
            else:
                flag = 2
        if flag == 2
            print('Test script should not be included in regression based on name')

    regressionDecisionLstName = []
    flag1 = 2
    for originalTestScriptName in testNameDf['UniqueTestNameValues']:
        for regressionScriptName in scriptsIncludedInRegressionNme:
            if regressionScriptName.lower() == originalTestScriptName.lower():
                flag1 = 1
                regressionDecisionLstName.append('Keyword foundin Test Script Name')
                break
            else:
                flag1 = 2
        if (flag1 == 2):
            regressionDecisionLstName.append('Keyword not found in Test Script Name')
            














