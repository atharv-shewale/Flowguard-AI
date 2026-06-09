import os
import joblib
import pymysql
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

app = FastAPI(
    title="Diabetes Progression Prediction API",
    description="API to predict diabetes progression using an AutoML model.",
    version="1.0.0"
)

# Load the model
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'model.pkl')
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    print(f"Warning: Could not load model at {MODEL_PATH}. Error: {e}")
    model = None

# Pydantic model for input validation
class FeaturesInput(BaseModel):
    age: float
    sex: float
    bmi: float
    bp: float
    s1: float
    s2: float
    s3: float
    s4: float
    s5: float
    s6: float
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 0.038,
                "sex": 0.050,
                "bmi": 0.061,
                "bp": 0.021,
                "s1": -0.044,
                "s2": -0.034,
                "s3": -0.043,
                "s4": -0.002,
                "s5": 0.019,
                "s6": -0.017
            }
        }
    }

class PredictionResponse(BaseModel):
    prediction: float

def get_db_connection():
    return pymysql.connect(
        host=os.environ.get("DB_HOST", "localhost"),
        user=os.environ.get("DB_USER", "root"),
        password=os.environ.get("DB_PASSWORD", ""),
        database=os.environ.get("DB_NAME", "diabetes_db")
    )

@app.post("/predict", response_model=PredictionResponse)
def predict(features: FeaturesInput):
    if not model:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
        
    try:
        input_list = [
            features.age, features.sex, features.bmi, features.bp,
            features.s1, features.s2, features.s3, features.s4,
            features.s5, features.s6
        ]
        
        # Predict
        prediction = float(model.predict([input_list])[0])
        
        # Save to database
        conn = get_db_connection()
        cursor = conn.cursor()
        query = """
            INSERT INTO predictions (age, sex, bmi, bp, s1, s2, s3, s4, s5, s6, prediction, source)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        values = tuple(input_list) + (prediction, 'fastapi')
        cursor.execute(query, values)
        conn.commit()
        cursor.close()
        conn.close()
        
        return PredictionResponse(prediction=prediction)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
