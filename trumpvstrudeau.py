#the classification problem is solved by svc and logistics regressor
#two different methods of vectorization are tested, tfidf and count vectorizer
#you can find better explanations in the google colab python notebook committed in the same repository



import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression

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


#classification applied with svc on tfidf data
svclassifier = SVC(kernel='rbf')
svclassifier.fit(t_train, y_train)
t_predsvc = svclassifier.predict(t_test)


#classification applied with svc on count vec data
svclassifier = SVC(kernel='rbf')
svclassifier.fit(c_train, y_train)
c_predsvc = svclassifier.predict(c_test)

#accuracy score of svc with count vectorizer
countsvcacc = accuracy_score(c_predsvc,y_test)
print(confusion_matrix(y_test,c_predsvc))
print(classification_report(y_test,c_predsvc))

#accuracy score of svc with tfidf vectorizer
tfidfsvmacc = accuracy_score(t_predsvc,y_test)
print(confusion_matrix(y_test,t_predsvc))
print(classification_report(y_test,t_predsvc))

#classification applied with logistic regression on tfidf vec data
logclassifier=LogisticRegression(random_state=0, solver='lbfgs')
logclassifier.fit(t_train, y_train)
t_predlog = logclassifier.predict(t_test)

#classification applied with logistic regression on count vec data
logclassifier=LogisticRegression(random_state=0, solver='lbfgs')
logclassifier.fit(c_train, y_train)
c_predlog = logclassifier.predict(c_test)


#accuracy score of logistic regression with count vectorizer
countlogacc = accuracy_score(c_predlog,y_test)
print(confusion_matrix(y_test,c_predlog))
print(classification_report(y_test,c_predlog))


#accuracy score of logistic regression with tfidf vectorizer
countlogacc = accuracy_score(t_predlog,y_test)
print(confusion_matrix(y_test,t_predlog))
print(classification_report(y_test,t_predlog))


#confusion matrix for logistic regressor
tlog_confmatrix = confusion_matrix(t_predlog,y_test)
clog_confmatrix = confusion_matrix(c_predlog,y_test)

#confusion matrix for svc
tsvc_confmatrix = confusion_matrix(t_predsvc,y_test)
csvc_confmatrix = confusion_matrix(c_predsvc,y_test)
