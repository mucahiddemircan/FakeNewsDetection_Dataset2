# Kütüphaneler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Veri yükleme
news = pd.read_csv('news.csv')

# Boş(NaN) olan satırları silme
news = news[news['text'].str.strip().ne('')]
news = news.dropna(subset=['text'])

#news = news.head(3000)

# Gereksiz tarih özniteliğinin ve boş olan eski title'ın silinmesi
news = news.drop(['title'], axis=1)
# Boş isme sahip kolonları sil
news = news.loc[:, ~news.columns.str.contains('^Unnamed')]

# Verilerin Ön İşlemesi
def word_operations(text):
    # Küçük harfe dönüştürülmesi
    text = text.lower()
    
    # Linklerin silinmesi
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    
    # HTML Taglerinin silinmesi
    text = re.sub(r'<.*?>', ' ', text)
    
    # Noktalama işaretlerinin silinmesi
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Sayıların silinmesi
    text = re.sub(r'\d', ' ', text)
    
    # Yeni satır karakterlerinin silinmesi (\n)
    text = re.sub(r'\n', ' ', text)
    
    # Sekme karakterlerinin silinmesi (\t)
    text = re.sub(r'\t', ' ', text)
    
    # Fazla boşlukların silinmesi
    text = re.sub(r'  ', ' ', text)
    
    # Özel karakterlerin silinmesi
    text = re.sub(r'[!@#$%^&*()_+-={}[]|:;"\'<>,.?/~\]', ' ', text)
    
    return text

# Tekrar eden kelimeleri kaldıran fonksiyon
def remove_duplicate_words(text):
    # Metni kelimelere böl
    words = text.split()
    # Benzersiz kelimeleri koru (kelimelerin sırasını koruyarak)
    unique_words = list(dict.fromkeys(words))
    # Kelimeleri yeniden birleştir
    return ' '.join(unique_words)

news['text'] = news['text'].apply(remove_duplicate_words)

# Stopwords kaldırma ve Lemmatization işlemleri için gerekli kütüphaneler
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
print(stop_words)

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# POS tag'i WordNet formatına dönüştüren yardımcı fonksiyon
def get_wordnet_pos(tag):
    if tag.startswith('J'):  # adjective(sıfat)
        return wordnet.ADJ
    elif tag.startswith('V'):   # verb(fiil)
        return wordnet.VERB
    elif tag.startswith('N'):   # noun(isim)
        return wordnet.NOUN
    elif tag.startswith('R'):   # adverb(zarf)
        return wordnet.ADV
    else:
        return None

# Etkisiz kelimelerin silinmesi
def remove_stop_words(text):
    return " ".join([word for word in str(text).split() if word not in stop_words])

# Stemming / Lemmatization (kelime köküne inilmesi)
def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    # Metni kelimelere ayırma
    words = word_tokenize(text)
    # Kelimelere POS tag ekleme
    words_with_pos = nltk.pos_tag(words)
    # Her bir kelime için lemmatization uygulama
    lemmatized_text = []
    for word, pos in words_with_pos:
        wordnet_pos = get_wordnet_pos(pos) or wordnet.NOUN  # Default olarak NOUN al
        lemmatized_text.append(lemmatizer.lemmatize(word, pos=wordnet_pos))
    return ' '.join(lemmatized_text)

print(news.head(10))
news['text'] = news['text'].apply(word_operations)
news['text'] = news['text'].apply(remove_stop_words)
news['text'] = news['text'].apply(lemmatize_text)
print(news.head(10))

# Feature Extraction (Öznitelik Çıkarımı)

# TF-IDF (Term Frequency-Inverse Document Frequency)(Kelimelerin sıklıklarını,önemini belirleme)
# Bag of Words,Yerleşim Vektörleri-Word Embeddings(Word2Vec,GloVe) gibi yöntemler de denenebilir.

x = news['text']
y = news['label']

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=0)

'''
#TfIdf
from sklearn.feature_extraction.text import TfidfVectorizer
vectorization = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)
'''

#Bag of Words
from sklearn.feature_extraction.text import CountVectorizer
vectorization = CountVectorizer(max_features=10000, ngram_range=(1, 2))  # Maksimum 10,000 kelime ve bigram kullanımı
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)

# Makine öğrenmesi modelininin seçimi ve uygulanması

# Lojistik Regresyon Sınıflandırması
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

LR = LogisticRegression()

LR.fit(xv_train, y_train)
prediction_LR = LR.predict(xv_test)
print("Lojistik Regresyon Doğruluğu:", LR.score(xv_test, y_test))
print(classification_report(y_test, prediction_LR))
print("Lojistik Regresyon Hata Matrisi:\n", confusion_matrix(y_test, prediction_LR))

sns.heatmap(confusion_matrix(y_test,prediction_LR),annot = True, cmap = 'coolwarm',fmt = '.1f')
plt.show()

# KNN Sınıflandırma
from sklearn.neighbors import KNeighborsClassifier

KNN = KNeighborsClassifier(n_neighbors=13, metric='cosine')  # k=5 olarak belirledik
KNN.fit(xv_train, y_train)
prediction_KNN = KNN.predict(xv_test)

print("KNN Doğruluğu:", KNN.score(xv_test, y_test))
print(classification_report(y_test, prediction_KNN))
print("KNN Hata Matrisi:\n", confusion_matrix(y_test, prediction_KNN))

# Yapay Sinir Ağları (ANN) Sınıflandırması
from sklearn.neural_network import MLPClassifier

ANN = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=0)
ANN.fit(xv_train, y_train)
prediction_ANN = ANN.predict(xv_test)

print("ANN Doğruluğu:", ANN.score(xv_test, y_test))
print(classification_report(y_test, prediction_ANN))
print("ANN Hata Matrisi:\n", confusion_matrix(y_test, prediction_ANN))

# Destek Vektör Makineleri (SVM) Sınıflandırması
from sklearn.svm import SVC

SVM = SVC(kernel='linear')  # Lineer kernel seçildi
SVM.fit(xv_train, y_train)
prediction_SVM = SVM.predict(xv_test)

print("SVM Doğruluğu:", SVM.score(xv_test, y_test))
print(classification_report(y_test, prediction_SVM))
print("SVM Hata Matrisi:\n", confusion_matrix(y_test, prediction_SVM))

# Karar Ağaçları Sınıflandırması
from sklearn.tree import DecisionTreeClassifier

DT = DecisionTreeClassifier()
DT.fit(xv_train, y_train)
prediction_DT = DT.predict(xv_test)

print("Decision Tree Doğruluğu:", DT.score(xv_test, y_test))
print(classification_report(y_test, prediction_DT))
print("Decision Tree Hata Matrisi:\n", confusion_matrix(y_test, prediction_DT))

