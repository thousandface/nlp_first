#1 去除html

from bs4 import BeautifulSoup
beautiful_text=BeautifulSoup().get_text()

#2，把非字母去掉
import re
letters_only=re.sub('[^a-zA-Z]',' ',beautiful_text)

#3,全部小写化
words=letters_only.lower().split()

#4,去除stopwords
from nltk.corpus import stopwords
stops=set(stopwords.words('english'))
meaningful_words=[w for w in words if not w in stops]

#5 高阶文字处理

#6,搞成一串string
#return(''.join(meaningful_words))

#把string 训练集变成 list of lists

#tokenizer=nltk.data.load('tokenizers/punkt/english.pickle)

from gensim.models import word2vec
num_feature=1000 #最多多少个不停的feature
min_word_count=10 #一个word出现多少次才被记录
num_wordkes=4 #多少thread一起跑
size=256  #vec的size
window=5#前后观察多长‘语境

#model=word2vec.Word2Vec(sentences,size=size,workers=num_wordkes,\
                        #size=num_feature,min_count=min_word_count,window=window)
#
# model.save('lol.save')#保存模型
# model=word2vec.Word2Vec.load('lol.sava')#日后load回来
#
# #几个常用的用法
# #woman+king-man=queen
# model.most_similar(positive=['woman','king'],nagative=['man'])
#
# #求两个词的相似度
# model.similarity('woman','man')
#
# model['computer']

