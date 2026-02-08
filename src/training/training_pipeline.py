import os
os.environ["PANDAS_ARROW_ENABLED"] = "0"

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split

from src.config.settings import DATASET_PATH, TEXT_COLUMN, LABEL_COLUMN, MODEL_PATH, VECTORIZER_PATH
from src.preprocessing.text_cleaner import clean_text
from src.features.vectorizer import build_vectorizer
from src.models.classifier import build_model
from src.evaluation.evaluator import evaluate_model


CATEGORY_MAP = {
    "POLITICS": "Politics",
    "WORLD NEWS": "Politics",
    "U.S. NEWS": "Politics",

    "THE WORLDPOST": "World",
    "WORLDPOST": "World",

    "BUSINESS": "Business",
    "MONEY": "Business",

    "SPORTS": "Sports",

    "TECH": "Technology",

    "SCIENCE": "Science",

    "ENTERTAINMENT": "Entertainment",
    "COMEDY": "Entertainment",

    "WELLNESS": "Lifestyle",
    "HEALTHY LIVING": "Lifestyle",
    "STYLE & BEAUTY": "Lifestyle",
    "TRAVEL": "Lifestyle",
    "FOOD & DRINK": "Lifestyle",
    "HOME & LIVING": "Lifestyle",
    "WEDDINGS": "Lifestyle",
    "PARENTING": "Lifestyle"
}


class TrainingPipeline:
    def run(self):
        df = pd.read_csv(DATASET_PATH)

        df[TEXT_COLUMN] = df[TEXT_COLUMN].astype(str).apply(clean_text)
        df[LABEL_COLUMN] = df[LABEL_COLUMN].map(CATEGORY_MAP)
        df = df.dropna(subset=[LABEL_COLUMN])

        print("Class distribution after mapping:")
        print(df[LABEL_COLUMN].value_counts())

        X_train, X_test, y_train, y_test = train_test_split(
            df[TEXT_COLUMN],
            df[LABEL_COLUMN],
            test_size=0.2,
            stratify=df[LABEL_COLUMN],
            random_state=42
        )

        vectorizer = build_vectorizer()
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)

        model = build_model()
        model.fit(X_train_vec, y_train)

        y_pred = model.predict(X_test_vec)
        acc, report, cm = evaluate_model(y_test, y_pred)

        print("\nAccuracy:", acc)
        print("\nClassification Report:\n", report)

        with open("artifacts/classification_report.txt", "w") as f:
            f.write(report)
            f.write("\nAccuracy: " + str(acc))

        joblib.dump(model, MODEL_PATH)
        joblib.dump(vectorizer, VECTORIZER_PATH)

        print("\n Model & Vectorizer saved in artifacts/")


if __name__ == "__main__":
    TrainingPipeline().run()
