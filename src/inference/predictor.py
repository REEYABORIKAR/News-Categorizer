import joblib
from src.config.settings import MODEL_PATH, VECTORIZER_PATH
from src.preprocessing.text_cleaner import clean_text

class Predictor:
    def __init__(self):
        self.model = joblib.load(MODEL_PATH)
        self.vectorizer = joblib.load(VECTORIZER_PATH)

    def predict_with_topk(self, text: str, k=3):
        text = clean_text(text)
        vec = self.vectorizer.transform([text])

        probs = self.model.predict_proba(vec)[0]
        classes = self.model.classes_

        top_idx = probs.argsort()[::-1][:k]
        topk = [(classes[i], float(probs[i])) for i in top_idx]

        label = topk[0][0]
        confidence = topk[0][1]

        return label, confidence, topk
