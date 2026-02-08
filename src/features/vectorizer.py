from sklearn.feature_extraction.text import TfidfVectorizer

def build_vectorizer():
    return TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=100000,
    stop_words="english",
    sublinear_tf=True
)


