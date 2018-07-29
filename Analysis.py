# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

#importing dataset
dataset = pd.read_csv('Employee_Comments.csv')
X=dataset['Question_or_Concern']

import urllib.parse
import urllib.request
import json
Pos=list()
Neg=list()
Sentiment=list()
keyword=list()

url = 'http://text-processing.com/api/sentiment/'
for words in X:
    values = {"text" : words}
    data = urllib.parse.urlencode(values).encode("utf-8")
    req = urllib.request.Request(url, data)
    response = urllib.request.urlopen(req)
    the_page = response.read()
    Positive=json.loads(the_page)
    a=Positive.get("probability").get("pos")
    b=Positive.get("probability").get("neg")
    c=Positive.get("probability").get("neutral")
    if b<a>c:
        Sentiment.append("pos")
    elif a<b>c:
        Sentiment.append("neg")
    else:
        Sentiment.append("pos")
  
 

dataset1 = pd.read_csv('Employee_Comments.csv', delimiter=',')
dataset['Sentiment'] =Sentiment  

#train['word_count'] = train['tweet'].apply(lambda x: len(str(x).split(" ")))
#train[['tweet','word_count']].head()


#cleaning
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus=[]
wordfreq=[]
for i in range(0,X.size):
    review=re.sub('[^a-zA-Z]',' ',dataset1['Question_or_Concern'][i])
    review=review.lower()
    review=review.split()
    ps=PorterStemmer()
    review=[ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    #review=' '.append(review)
    corpus.append(review)
   # for words in review:
     #    wordfreq.append(review.count(words))
   # dataset1[words]

columnList=[] 
for x in corpus:
    for y in x:
        dataset1[y]=0
        columnList.append(y)

     
for i in range(0,X.size):
    review=re.sub('[^a-zA-Z]',' ',dataset1['Question_or_Concern'][i])
    review=review.lower()
    review=review.split()
    ps=PorterStemmer()
    review=[ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    for words in review:
        for xyz in columnList:
            if words==xyz:
               dataset1.at[i,xyz]=dataset1.at[i,xyz]+1

columns=[ 'floor', 'access' ,'ramp', 'cannot', 'use', 'get',
 'part', 'offic' ,'peopl', 'make', 'room' ,'lift' ,'enough', 'cubicl',
'work', 'take' ,'care',
 'mani' ,'even', 'call',  'need', 
 'want', 'alway', 'feel' ,'think' ,'put' ,'much', 'effort',
'given' , 'pass', 'allow',
'tri', 'tool', 'date' ,'meet' ,
 'present', 'way', 'swear',  'give', 'custom']
col=['EmployeeID','Question_or_Concern','Anonymous','Timestamp','Sentiment','keyword','keyword_count']
dataset_final=dataset1
dataset1=dataset1.drop(col,axis=1) 
dataset1=dataset1.drop(columns,axis=1) 
keyword_count=[]
dataset['keyword_count']=dataset1.max(axis=1)
dataset['keyword']=dataset1.apply(lambda x:x.argmax(),axis=1)
dataset['keyword']=dataset['keyword'].replace('flexibl','flexible')
dataset['keyword']=dataset['keyword'].replace('famili','family')
dataset['keyword']=dataset['keyword'].replace('colleagu','colleague')
dataset['keyword']=dataset['keyword'].replace('colleaug','colleague')
dataset['keyword']=dataset['keyword'].replace('tradit','traditional')
dataset['keyword']=dataset['keyword'].replace('opportun','opportunity')
dataset['keyword']=dataset['keyword'].replace('includ','include')
dataset.to_csv('Employee_Comments1.csv',sep=',')
#for i in range(0,X.size):
 #   if dataset.at(i,'keyword')=='flexibl':
  #      dataset.set_value(i,'keyword','flexible')
        
#from sklearn.feature_extraction.text import CountVectorizer
#cv=CountVectorizer(max_features=1500)
#XT=cv.fit_transform(corpus).toarray()
#y=dataset1
