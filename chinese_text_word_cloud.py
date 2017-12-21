import warnings
warnings.filterwarnings('ignore')
import jieba
import numpy
import codecs # condecs提供open方法来指定打开文件的语言编码，它会在读取的时候自动转换为内部unicode
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from wordcloud import WordCloud

df=pd.read_csv('finance_news.csv',encoding='utf-8')
df=df.dropna()
content=df.content.values.tolist()
segment=[]
for line in content:
    try:
        segs=jieba.lcut(line)
        for seg in segs:
            if len(seg)>1 and seg!='\r\n':
                segment.append(seg)
    except:
        print(line)
        continue

#去掉停用词
word_df=pd.DataFrame({'segment':segment})

stopword=pd.read_csv('stopwords.txt',index_col=False,quoting=3,sep='\t',names=['stopword'],encoding='utf-8') #quoting=3全不引用
word_df=word_df[~word_df.segment.isin(stopword)]

#统计词频
word_stat=word_df.groupby(['segment'])['segment'].agg({'计数':numpy.size})
word_stat=word_stat.reset_index().sort_values(['计数'],ascending=False)  #从大到小排  reset_index把其拉平
print(word_stat.head())

#做词云
wordcloud=WordCloud(font_path="simhei.ttf",background_color="white",max_font_size=80)
word_frequence = {x[0]:x[1] for x in word_stat.head(1000).values}
wordcloud=wordcloud.fit_words(word_frequence)
plt.imshow(wordcloud)


#用特定的图片模板显示
from scipy.misc import imread
matplotlib.rcParams['figure','figsize']=(15.0,15.0)
from wordcloud import WordCloud,ImageColorGenerator
bimg=imread('entertainment.jpeg')
wordcloud=WordCloud(background_color='white',mask=bimg,font_path='simhei.ttf',max_font_size=200)
word_frequence={x[0]:x[1] for x in word_stat.head(1000).values}
wordcloud=wordcloud.fit_words(word_frequence)
bimgColors=ImageColorGenerator(bimg)
plt.axis('off')
plt.imshow(wordcloud.recolor(color_func=bimgColors))

