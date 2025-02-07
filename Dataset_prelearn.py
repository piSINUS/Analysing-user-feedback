import pandas as pd 
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from transformers import BertTokenizer, TFBertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


nltk.download('punkt')  # Для токенизации
nltk.download('stopwords')  # Стоп-слова
nltk.download('wordnet') # Для лемматизации
nltk.download('punkt_tab')

df = pd.read_csv("DisneylandReviews.csv", encoding='latin-1')
# print(df.head())
# # print(df.info())
# # print(df.isnull().sum())
# print(df['rating'])

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

def label_sentiment(rating):#разметка данных
    if rating >= 4:
        return "positive"
    elif rating == 3:
        return "neutral"
    else:
        return "negative"


df["sentiment"] = df['rating'].apply(label_sentiment)
df['cleaned_review'] = df['review_text'].apply(clean_text)  # Очистка текста
df['tokenized_review'] = df['cleaned_review'].apply(tokenize_and_remove_stopwords)  # Токенизация и удаление стоп-слов
df['lemmatized_review'] = df['tokenized_review'].apply(lemmatize_words)  # Лемматизация
df['lemmatized_review'] = df['lemmatized_review'].apply(lambda x: ' '.join(x))  # Обратно в строку


words = " ".join(df["lemmatized_review"].dropna())

if len(words) > 0:
    wordcloud = WordCloud(width=800, height=400, background_color="white")
    wordcloud.generate(words)  # Отдельный вызов

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()
else:
    print("Ошибка: текст для WordCloud пуст!")

word_freq = Counter(words.split())
common_words = word_freq.most_common(20)

# plt.barh([word[0] for word in common_words], [word[1] for word in common_words])
# plt.xlabel("Частота")
# plt.ylabel("Слово")
# plt.title("Топ-20 слов в отзывах")
# plt.show()
# Преобразование текста в векторный формат
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
tokens = tokenizer(df["lemmatized_review"].tolist(),padding = True)

# Разделение данных на обучающую и тестовую выборки:
X_train, X_test, y_train, y_test = train_test_split(X, df["sentiment"], test_size=0.2, random_state=42)

encoder