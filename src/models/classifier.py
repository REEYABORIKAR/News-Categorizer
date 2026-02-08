from sklearn.linear_model import LogisticRegression

def build_model():
    return LogisticRegression(
    max_iter=3000,
    n_jobs=-1,
    class_weight="balanced"
)

