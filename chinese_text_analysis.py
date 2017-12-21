import jieba.analyse as analyse
import pandas as pd
df=pd.read_csv('finance_news.csv',encoding='utf-8')
df=df.dropna()
lines=df.content.values.tolist()
content=''.join(lines)
#tf-idf取关键词
print(' '.join(analyse.extract_tags(content,topK=10,withWeight=False,allowPOS=())))

#textRank算法关键词分析

df=pd.read_csv('finance_news.csv',encoding='utf-8')
df=df.dropna()
lines=df.content.values.tolist()
content=''.join(lines)
print(' '.join(analyse.textrank(content,topK=10,withWeight=False,allowPOS=('ns','n','vn'))))

#LAD主题模型

from gensim import corpora,models,similarities
import gensim
#载入停止词
stopword=pd.read_csv('stopwords.txt',index_col=False,quoting=3,sep='\t',names=['stopword'],encoding='utf-8')
stopword=stopword['stopword'].values

#转换为合适的格式
import jieba
import pandas as pd
df=pd.read_csv('finance_news.csv',encoding='utf-8')
df=df.dropna()
lines=df.content.values.tolist()
sentence=[]
for line in lines:
    try:
        segs=jieba.lcut(line)
        segs=filter(lambda x:len(x)>1,segs)
        segs=filter(lambda x:x not in stopword,segs)
        sentence.append(segs)
    except:
        print(line)
        continue

#词袋模型
dictionary=corpora.Dictionary(sentence)
corpus=[dictionary.doc2bow(sentence) for sentence in sentence] #将每一行转换为计算机能看懂的模式[(91, 1),（105,5）]

#lda建模
lda=gensim.models.LdaModel(corpus=corpus,id2word=dictionary,num_topics=10)
#查下第三号分类，其中最常出现的单词
print(lda.print_topic(3,topn=5))

#把一个文本分为多个主题 ，每个主题用多个词表示
#把所有主题打印出来
for k,topic in lda.print_topic(num_topics=10,num_words=8):
    print(topic)