from src.inference.predictor import Predictor

if __name__=="__main__":
    predictor=Predictor()

    while True:
        text=input("Enter news text (or 'exit): ")
        if text.lower()=="exit":
            break

        label,confidence=predictor.predict(text)
        print(f"Predicted Category: {label} | Confidence: {confidence:.2f}")