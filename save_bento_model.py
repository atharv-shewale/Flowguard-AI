import bentoml
import joblib

if __name__ == "__main__":
    print("Loading joblib model...")
    try:
        model = joblib.load("model.pkl")
        
        print("Saving to BentoML model store...")
        bento_model = bentoml.sklearn.save_model("diabetes_model", model)
        print(f"Successfully saved to BentoML: {bento_model}")
    except Exception as e:
        print(f"Error saving to BentoML: {e}")
