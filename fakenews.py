import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
df=pd.read_csv('C:\\Users\\Administrator\\Downloads\\news.csv')
df.head(5)
print(df)
labels = df.label
labels.head()
x_train,x_test,y_train,y_test=train_test_split(df['text'],labels,test_size=0.2,random_state=7)
tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)
tfidf_train=tfidf_vectorizer.fit_transform(x_train) 
tfidf_test=tfidf_vectorizer.transform(x_test)
pac=PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train,y_train)
y_pred=pac.predict(tfidf_test)
score=accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(score*100,2)}%')
array=confusion_matrix(y_test,y_pred, labels=['FAKE','REAL'])
print(array)
df=pd.read_csv('C:\\Users\\Administrator\\Downloads\\news.csv')
df.describe(include=np.object)
print(df.groupby('label').count())
plt.hist(df['label'], facecolor='green',edgecolor='black',linewidth=1.2)
plt.show()
a= plt.hist(dta['label'])
plt.ylim([3150, 3180])
