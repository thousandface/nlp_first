from nltk.classify import NaiveBayesClassifier

s1='this is a good book'
s2='this is a awesome book'
s3='this is a bad book'
s4='this is a terribe book'

def preprocess(s):
    return {word:True for word in s.lower().split()}

training_data=[[preprocess(s1),'pos'],
               [preprocess(s2),'pos'],
               [preprocess(s3),'neg'],
               [preprocess(s4),'neg'],
               ]
model=NaiveBayesClassifier.train(training_data)
print(model.classify(preprocess('this is a good book')))