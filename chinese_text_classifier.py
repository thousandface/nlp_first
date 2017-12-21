import jieba
import pandas as pd
df_tec=pd.read_csv('technology_news.csv',encoding='utf-8')
df_tec=df_tec.dropna()

df_home=pd.read_csv('home_news.csv',encoding='utf-8')
df_home=df_home.dropna()
df_inter=pd.read_csv('international_news.csv',encoding='utf-8')
df_inter=df_inter.dropna()
df_fin=pd.read_csv('finance_news.csv',encoding='utf-8')
df_fin=df_fin.dropna()

technology = df_tec.content.values.tolist()[1000:21000]
international = df_inter.content.values.tolist()[1000:21000]
finance = df_fin.content.values.tolist()[:20000]
home = df_home.content.values.tolist()[:20000]

#分词与文本处理

#停用词
stopword=pd.read_csv('stopwords.txt',index_col=False,quoting=3,names=['stopword'],encoding='utf-8')
stopword=stopword['stopword'].values

#去掉停用词

def preprocess_text(conten_lines,sentence,catefory):
    for line in conten_lines:
        try:
            segs=jieba.lcut(line)
            segs=filter(lambda x:len(x)>1,segs)
            segs=filter(lambda x:x not in stopword,segs)
            sentence.append((' '.join(segs),catefory))
        except:
            print(line)
            continue

#生成训练数据
sentences=[]
preprocess_text(technology,sentences,'technology')
preprocess_text(international,sentences,'international')
preprocess_text(home,sentences,'home')
preprocess_text(finance,sentences,'finance')

#生成训练集 打乱下顺序
import random
random.shuffle(sentences)
for sen in sentences[:10]:
    print(sen)
#
#分数据集
from sklearn.cross_validation import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1234)

#下一步在降噪数据上抽取有用的特征，对文本抽取词袋模型
from sklearn.feature_extraction.text import CountVectorizer
vec=CountVectorizer(
    analyzer='word', #基于词分析
    max_features=10000,
)
vec.fit(x_train)
def get_ferture(x):
    vec.transform(x)

# #import分类器进行训练
from sklearn.naive_bayes import MultinomialNB
classifer=MultinomialNB()
classifer.fit(vec.transform(x_train),y_train)



print(classifer.score(vec.transform(x_test),y_test))



#分层交叉验证
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import accuracy_score,precision_score
import numpy as np

def stratifiedkfold_cv(x,y,clf_class,shuffle=True,n_flods=5,**kwargs):
    stratifiedk_fold=StratifiedKFold(y,n_flods=n_flods,shuffle=shuffle)
    y_pred=y[:]
    for train_index,text_index in stratifiedk_fold:
        x_train,x_test=x[train_index],x[text_index]
        y_train=y[train_index]
        clf=clf_class(**kwargs)
        clf.fit(x_train,y_train)
        y_pred[text_index]=clf.predict(x_test)
    return y_pred

NB=MultinomialNB
print(precision_score(y,stratifiedkfold_cv(vec.transform(x),np.array(y),NB),average='macro'))