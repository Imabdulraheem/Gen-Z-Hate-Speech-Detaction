# 1. Import Libraries
import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import warnings
warnings.filterwarnings('ignore')

# NLP tools
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# 2. Load Dataset
csv_path = './train.csv'
tweet_df = pd.read_csv(csv_path)
tweet_df = tweet_df[['tweet', 'label']]

# 3. Preprocessing Functions
def data_processing(tweet):
    tweet = tweet.lower()
    tweet = re.sub(r"https\S+|www\S+http\S+", '', tweet)
    tweet = re.sub(r'\@w+|\#','', tweet)
    tweet = re.sub(r'[^\w\s]','', tweet)
    tweet = re.sub(r'ð','', tweet)
    tweet_tokens = word_tokenize(tweet)
    filtered = [w for w in tweet_tokens if w not in stop_words]
    return " ".join(filtered)

def lemmatizing(tweet):
    return " ".join([lemmatizer.lemmatize(word) for word in tweet.split()])

# 4. Apply Preprocessing
tweet_df['tweet'] = tweet_df['tweet'].apply(data_processing).apply(lemmatizing)
tweet_df = tweet_df.drop_duplicates('tweet')

# 5. TF-IDF Vectorization
vectorizer = TfidfVectorizer(ngram_range=(1, 3))
X = vectorizer.fit_transform(tweet_df['tweet'])
y = tweet_df['label']

# 6. Train-Test Split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Model Training with GridSearchCV
param_grid = {'C':[100, 10, 1.0, 0.1, 0.01], 'solver':['newton-cg', 'lbfgs', 'liblinear']}
grid = GridSearchCV(LogisticRegression(), param_grid, cv=5)
grid.fit(x_train, y_train)

# 8. Evaluation
y_pred = grid.predict(x_test)
print("Accuracy: {:.2f}%".format(accuracy_score(y_pred, y_test) * 100))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(confusion_matrix=cm).plot()

# 9. Predict and Explain Function
def process_input_tweet(tweet):
    tweet = tweet.lower()
    tweet = re.sub(r"https\S+|www\S+http\S+", '', tweet)
    tweet = re.sub(r'\@w+|\#','', tweet)
    tweet = re.sub(r'[^\w\s]','', tweet)
    tweet = re.sub(r'ð','', tweet)
    tokens = word_tokenize(tweet)
    filtered = [w for w in tokens if w not in stop_words]
    lemmatized = [lemmatizer.lemmatize(w) for w in filtered]
    return " ".join(lemmatized)

def classify_tweet(tweet_text, vectorizer, model, top_n=5):
    clean_tweet = process_input_tweet(tweet_text)
    vectorized_tweet = vectorizer.transform([clean_tweet])
    prediction = model.predict(vectorized_tweet)[0]
    prediction_proba = model.predict_proba(vectorized_tweet)[0]
    label = "Hate Speech" if prediction == 1 else "Not Hate Speech"

    feature_array = np.array(vectorizer.get_feature_names_out())
    tweet_vector = vectorized_tweet.toarray().flatten()
    nonzero_indices = tweet_vector.nonzero()[0]
    weights = model.coef_[0]
    contributions = [(feature_array[i], tweet_vector[i] * weights[i]) for i in nonzero_indices]
    contributions = sorted(contributions, key=lambda x: abs(x[1]), reverse=True)
    top_features = contributions[:top_n]

    print(f"\nTweet: {tweet_text}")
    print(f"Cleaned: {clean_tweet}")
    print(f"\nPrediction: {label}")
    print(f"Confidence: {prediction_proba[prediction]*100:.2f}%")
    print("\nTop contributing features:")
    for word, score in top_features:
        print(f"  - {word}: {'+' if score >= 0 else ''}{score:.4f}")

# 10. Try on Example Tweet
test_tweet = "I hate those people, they are the worst!"
flag = True
while flag:
    message = input("Enter a tweet to classify (or type 'exit' to quit): ")
    if message.lower() == 'exit':
        flag = False
    else:
        test_tweet = message
        classify_tweet(test_tweet, vectorizer, grid.best_estimator_)
