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



