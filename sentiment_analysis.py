# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud

# Download necessary NLTK datasets
nltk.download('punkt')
nltk.download('stopwords')

# Load the dataset
data = pd.read_csv('twitter_training.csv')
data = pd.read_csv('twitter_validation.csv')

# Display the first few rows
print(data.head())

# Preprocessing the tweet text
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Tokenize and remove stopwords
    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    return " ".join(filtered_tokens)

# Apply text preprocessing
data['cleaned_text'] = data['text'].apply(preprocess_text)

# Encode sentiment labels
sentiment_mapping = {'positive': 1, 'neutral': 0, 'negative': -1}
data['sentiment_label'] = data['sentiment'].map(sentiment_mapping)

# Split the data into features and target
X = data['cleaned_text']
y = data['sentiment_label']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Convert text data into numerical data using TF-IDF Vectorizer
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Train a Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_tfidf)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Visualize sentiment distribution
sns.countplot(x='sentiment', data=data, palette='viridis')
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()

# Create a WordCloud for positive sentiments
positive_text = " ".join(data[data['sentiment_label'] == 1]['cleaned_text'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(positive_text)

# Plot the WordCloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('WordCloud for Positive Sentiment Tweets')
plt.show()

# Visualize the most important words in the dataset
top_words = pd.DataFrame(tfidf.get_feature_names_out(), columns=['Word'], data=model.coef_[0])
top_words_sorted = top_words.sort_values(by='Word', ascending=False).head(20)

# Plot top 20 most important words
plt.figure(figsize=(10, 6))
sns.barplot(x=top_words_sorted['Word'], y=top_words_sorted['Word'])
plt.title('Top 20 Important Words')
plt.xlabel('Words')
plt.ylabel('Importance')
plt.xticks(rotation=90)
plt.show()