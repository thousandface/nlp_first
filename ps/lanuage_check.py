in_f=open('./data/data.csv','rb')
lines=in_f.readlines()
in_f.close()
dataset=[(line.strip()[:-3],line.strip()[-2:])for line in lines]
#print(dataset[:5])

#把数据集分为训练集和测试集
import sklearn
train_data=dataset[100:]
test_data=dataset[:100]
x_test,y_test=zip(*test_data)
x_train,y_train=zip(*train_data)

#去噪声
import re
def remove_noise(document):
    noise_pattern=re.compile('|'.join(['http\S+','\@\w+','\#\w+']))
    clean_text=re.sub(noise_pattern,'',document)
    return clean_text.strip()

#print(remove_noise("Trump images are now more popular than cat gifs. @trump #trends http://www.trumptrends.html"))

#在降噪声的数据上抽取 1-gram 和2-gram
from sklearn.feature_extraction.text import CountVectorizer
vec=CountVectorizer(
    lowercase=True,  #变小写
    analyzer='char_wb', #由字符分
    ngram_range=(1,2),#用1-gram和2
    max_features=1000, #抽取1000个
    preprocessor=remove_noise  #先预处理
)
vec.fit(x_train)
def get_features(x):
    vec.transform(x)

#把分类器弄进来训练
from sklearn.naive_bayes import MultinomialNB
Classifier=MultinomialNB()
Classifier.fit(vec.transform(x_train),y_train)

pred=Classifier.predict(vec.transform('i am student'))

print(pred)