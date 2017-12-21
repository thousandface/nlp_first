import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import MultinomialNB

class TextClassifier():
    def __int__(self,classifier=MultinomialNB()):
        self.classifier=classifier
        self.vectorizer=CountVectorizer(analyzer='word',ngram_range=(1,4),max_features=8000)
    def features(self,x):
        return self.vectorizer.transform(x)

    def fit(self,x,y):
        self.vectorizer.fit(x)
        self.classifier.fit(self.features(x),y)
    def predict(self,x):
        return self.classifier.predict(self.features([x]))
    def score(self,x,y):
        return self.classifier.score(self.features(x),y)

#svm做文本分类
# from sklearn.svm import SVC
# svm = SVC(kernel='linear')
# svm.fit(vec.transform(x_train), y_train)
# svm.score(vec.transform(x_test), y_test)

