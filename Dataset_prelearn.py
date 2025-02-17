import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Загружаем датасет
df = pd.read_csv("DisneylandReviews.csv")

# Проверяем, есть ли нужная колонка
if "review_text" not in df.columns:
    raise KeyError("Колонка 'review_text' не найдена в датасете")

# Загружаем словарь VADER
nltk.download("vader_lexicon")
sia = SentimentIntensityAnalyzer()

# Размечаем отзывы автоматически
def classify_sentiment(text):
    score = sia.polarity_scores(str(text))["compound"]  # Анализируем тональность
    return 1 if score > 0 else 0  # Позитив = 1, Негатив = 0

df["sentiment"] = df["review_text"].apply(classify_sentiment)

# Проверяем разметку
print(df[["review_text", "sentiment"]].head(10))

# Преобразуем текст в TF-IDF векторы
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["review_text"])
y = df["sentiment"]

# Разбиваем на обучающую и тестовую выборку
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучаем логистическую регрессию
model = LogisticRegression()
model.fit(X_train, y_train)

# Делаем предсказания
y_pred = model.predict(X_test)

# Оцениваем точность
print("Accuracy:", accuracy_score(y_test, y_pred))

#  Провекрка 
new_reviews = ["I love this product!", "This is the worst experience ever."]
X_new = vectorizer.transform(new_reviews)
predictions = model.predict(X_new)

for review, pred in zip(new_reviews, predictions):
    print(f"Review: {review} -> Sentiment: {'Positive' if pred == 1 else 'Negative'}")
