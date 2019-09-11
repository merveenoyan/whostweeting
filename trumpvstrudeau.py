#the classification problem is solved by svm
#two different methods of vectorization are tested
#being, tfidf and count vectorization
#will be improved as I better myself in NLP


import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

#importing data
tweetsdf = pd.read_table('/Users/Merveilleux/Desktop/tweets.txt', sep=',', names=('ID', 'Author', 'tweet'))

#removed first row consisting of labels 
tweetsdf=tweetsdf.iloc[1:]


#separating data for train and test
#we will predict author from tweets 
y=tweetsdf['Author']
x=tweetsdf['tweet']
x_train, x_test, y_train, y_test =train_test_split(x,y,test_size=0.33, random_state=50)

#tfidf vectorizer helps us get rid of words that appear a lot yet 
#doesn't give idea about the content of the text by giving them less value
tvec= TfidfVectorizer(stop_words="english", ngram_range=(1,2), max_df=0.9, min_df=0.05)
t_train=tvec.fit_transform(x_train)
t_test=tvec.fit_transform(x_test)

#count vectorizer helps cleaning the text by implementing tokens & counting
#with n gram we count unigram, bigram and trigram words
#max df and min df in extracts corpus specific stopwords
cvec = CountVectorizer(stop_words="english",ngram_range=(1,2), max_df=0.9, min_df=0.05)
c_train=cvec.fit_transform(x_train)
c_test=cvec.fit_transform(x_test)


#svm applied on model based on tfidf
svclassifier = SVC(kernel='linear')
svclassifier.fit(t_train, y_train)
t_pred = svclassifier.predict(t_test)

#accuracy test and confusion matrix
tfidfacc = accuracy_score(t_pred,y_test)
print(confusion_matrix(y_test,t_pred))
print(classification_report(y_test,t_pred))


#svm applied on model based on count vectorizer
svclassifier = SVC(kernel='linear')
svclassifier.fit(c_train, y_train)
c_pred = svclassifier.predict(c_test)

#accuracy test and confusion matrix
countacc = accuracy_score(c_pred,y_test)
print(confusion_matrix(y_test,c_pred))
print(classification_report(y_test,c_pred))



t_confmatrix = confusion_matrix(t_pred,y_test)
c_confmatrix = confusion_matrix(c_pred,y_test)

