from langdetect import detect, LangDetectException

def detect_language(text):
    """Detects the language of a given text using langdetect."""
    try:
        lang = detect(text)
        return lang
    except LangDetectException as e:
        return f"Error: {e}"


if __name__ == "__main__":
    text = input("Enter text: ")
    detected_language = detect_language(text)
    print(f"Detected language: {detected_language}")


import nltk #Natural Language Toolkit (for tokenization, etc.)
from sklearn.naive_bayes import MultinomialNB #Example Model
from sklearn.feature_extraction.text import TfidfVectorizer #Example Feature Extractor


vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(training_texts) 
clf = MultinomialNB()
clf.fit(X, training_languages)

new_text = input("Enter text: ")
X_new = vectorizer.transform([new_text])
predicted_language = clf.predict(X_new)[0]
print(f"Predicted language: {predicted_language}")

