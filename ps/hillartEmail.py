import numpy as np
import pandas as pd
import re

df=pd.read_csv('./data/HillaryEmails.csv')
df=df[['Id','ExtractedBodyText']].dropna() #去掉nan值
#文本预处理
def clean_email_text(text):
    text=text.replace('\n',' ')#去掉换行
    text=re.sub(r'-',' ',text)#把 "-" 的两个单词，分开。（比如：july-edu ==> july edu）
    text=re.sub(r'\d+/\d+/\d',' ',text)#日期，对主体模型没什么意义
    text = re.sub(r"[0-2]?[0-9]:[0-6][0-9]", "", text) #时间，没意义
    text = re.sub(r"[\w]+@[\.\w]+", "", text) #邮件地址，没意义
    text = re.sub(r"/[a-zA-Z]*[:\//\]*[A-Za-z0-9\-_]+\.+[A-Za-z0-9\.\/%&=\?\-_]+/i", "", text) #网址，没意义
    pure_text = ''
    for letter in text:
        if letter.isalpha() or letter==' ':
            pure_text+=letter
    text=' '.join(word for word in pure_text.split() if len(word)>1)
    return text
#现在我们新建一个colum，并把我们的方法跑一遍：

docs=df['ExtractedBodyText']
docs=docs.apply(lambda s:clean_email_text(s))

#print(docs.head(1).values)
doclist=docs.values #把df转换为numpy

#Lda模型构造
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim') #在gensim前面加上防止错误

from gensim import corpora,models,similarities
import gensim

#停止词列表
stoplist = ['very', 'ourselves', 'am', 'doesn', 'through', 'me', 'against', 'up', 'just', 'her', 'ours',
            'couldn', 'because', 'is', 'isn', 'it', 'only', 'in', 'such', 'too', 'mustn', 'under', 'their',
            'if', 'to', 'my', 'himself', 'after', 'why', 'while', 'can', 'each', 'itself', 'his', 'all', 'once',
            'herself', 'more', 'our', 'they', 'hasn', 'on', 'ma', 'them', 'its', 'where', 'did', 'll', 'you',
            'didn', 'nor', 'as', 'now', 'before', 'those', 'yours', 'from', 'who', 'was', 'm', 'been', 'will',
            'into', 'same', 'how', 'some', 'of', 'out', 'with', 's', 'being', 't', 'mightn', 'she', 'again', 'be',
            'by', 'shan', 'have', 'yourselves', 'needn', 'and', 'are', 'o', 'these', 'further', 'most', 'yourself',
            'having', 'aren', 'here', 'he', 'were', 'but', 'this', 'myself', 'own', 'we', 'so', 'i', 'does', 'both',
            'when', 'between', 'd', 'had', 'the', 'y', 'has', 'down', 'off', 'than', 'haven', 'whom', 'wouldn',
            'should', 've', 'over', 'themselves', 'few', 'then', 'hadn', 'what', 'until', 'won', 'no', 'about',
            'any', 'that', 'for', 'shouldn', 'don', 'do', 'there', 'doing', 'an', 'or', 'ain', 'hers', 'wasn',
            'weren', 'above', 'a', 'at', 'your', 'theirs', 'below', 'other', 'not', 're', 'him', 'during', 'which']

#人工分词  分词的意义在于，把我们的长长的字符串原文本，转化成有意义的小元素：
texts=[[word for word in doc.lower().split() if word not in stoplist]for doc in doclist]

#建立语料库  把每一个单词用一个数字index指代  并把原文本变成一条长长的数组
dictionary=corpora.Dictionary(texts) #得出一个词对照表
corpus=[dictionary.doc2bow(text)for text in texts] #把其转换为数字  第一个是index 第二个是出现了几次

lda=gensim.models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=20)
print(lda.print_topic(9,topn=5)) #打印前五个主题


