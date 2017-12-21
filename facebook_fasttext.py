import jieba
import pandas as pd
import random

cate_dic={'technology':1,'international':2,'home':3,'finance':4}
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
            sentence.append('__label__'+str(catefory)+','+' '.join(segs))
        except:
            print(line)
            continue

#生成训练数据
sentences=[]
preprocess_text(technology,sentences,cate_dic['technology'])
preprocess_text(international,sentences,cate_dic['international'])
preprocess_text(home,sentences,cate_dic['home'])
preprocess_text(finance,sentences,cate_dic['finance'])
random.shuffle(sentences)
print('writing data to fasttext format...')
out=open('train_data.txt','w')
for sentences in sentences:
    out.write(sentences.encode('UTF-8')+b'\n')
print('done')

import fasttext
#调用fastText训练生成模型
classifier=fasttext.supervised('trian_data.txt','classifier.model',label_prefix='__label__')

#对模型效果进行评估
result=classifier.test('train_data.txt')
print('p:'+result.precision)
print('r:'+result.recall)
print('e:'+result.nexamples)

#实际预测
label_to_cate={1:'technology',2:'international',3:'home',4:'finance'}
texts=['中国队 率先 打破 场上 僵局 获得 成功']
labels=classifier.predict(texts)
print(labels)
print(label_to_cate[int(labels[0][0])])

labels=classifier.predict_proba(texts)

