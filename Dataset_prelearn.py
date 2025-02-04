import pandas as pd 
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')  # Для токенизации
nltk.download('stopwords')  # Стоп-слова
nltk.download('wordnet') # Для лемматизации

df = pd.read_csv("dataset.csv")
# print(df.head())
# print(df.info())
# print(df.isnull().sum())

def clean_text(text): #Очищение текста от HTML-тегов, эмодзи и лишних символов.
    text = text.lower()
    text = re.sub(r'<,.*?>','',text)
    text = re.sub(r'[^a-zA-Zа-яА-Я0-9\s]', '', text)
    text = re.sub(r'\d+', '', text)
    return text

def tokenize_and_remove_stopwords(text): #Функция токенизации и удаления стоп-слов
    stop_words = set(stopwords.words('russian') + stopwords.words('english'))  # Объединяем списки стоп-слов
    words = word_tokenize(text)  # Токенизация текста
    words = [word for word in words if word not in stop_words]  # Убираем стоп-слова
    return words

lemmatizer = WordNetLemmatizer()

def lemmatize_words(words):
    return [lemmatizer.lemmatize(word) for word in words]

df['cleaned_review'] = df['review_text'].apply(clean_text)  # Очистка текста
df['tokenized_review'] = df['cleaned_review'].apply(tokenize_and_remove_stopwords)  # Токенизация и удаление стоп-слов
df['lemmatized_review'] = df['tokenized_review'].apply(lemmatize_words)  # Лемматизация
df['lemmatized_review'] = df['lemmatized_review'].apply(lambda x: ' '.join(x))  # Обратно в строку

print(df[['review_text', 'cleaned_review', 'lemmatized_review']].head())