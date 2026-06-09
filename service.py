import bentoml

@bentoml.service
class DiabetesService:
    # Get the model reference from the BentoML store
    model_ref = bentoml.models.get("diabetes_model:latest")

    def __init__(self):
        # Load the scikit-learn model into memory
        import bentoml.sklearn
        self.model = bentoml.sklearn.load_model(self.model_ref)

    @bentoml.api
    def predict(self, features: list) -> dict:
        """
        Expects input in the format: {"features": [age, sex, bmi, bp, s1, s2, s3, s4, s5, s6]}
        """
        if len(features) != 10:
            return {"error": "Expected exactly 10 feature values."}
            
        result = self.model.predict([features])
        return {"prediction": float(result[0])}

