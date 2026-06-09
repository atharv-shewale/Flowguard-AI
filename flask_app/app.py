import os
import joblib
import pymysql
from flask import Flask, request, render_template

app = Flask(__name__)

# Load the model (assumes model.pkl is one directory up, in the root project folder)
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'model.pkl')
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    print(f"Warning: Could not load model at {MODEL_PATH}. Error: {e}")
    model = None

def get_db_connection():
    return pymysql.connect(
        host=os.environ.get("DB_HOST", "localhost"),
        user=os.environ.get("DB_USER", "root"),
        password=os.environ.get("DB_PASSWORD", ""),
        database=os.environ.get("DB_NAME", "diabetes_db")
    )

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return render_template('result.html', error="Model is not loaded. Please train the model first.")

    try:
        # Extract features from form
        features = [
            float(request.form['age']),
            float(request.form['sex']),
            float(request.form['bmi']),
            float(request.form['bp']),
            float(request.form['s1']),
            float(request.form['s2']),
            float(request.form['s3']),
            float(request.form['s4']),
            float(request.form['s5']),
            float(request.form['s6'])
        ]
        
        # Predict using FLAML model
        prediction = float(model.predict([features])[0])
        
        # Save to database
        conn = get_db_connection()
        cursor = conn.cursor()
        query = """
            INSERT INTO predictions (age, sex, bmi, bp, s1, s2, s3, s4, s5, s6, prediction, source)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        values = tuple(features) + (prediction, 'flask_ui')
        cursor.execute(query, values)
        conn.commit()
        cursor.close()
        conn.close()
        
        return render_template('result.html', prediction=round(prediction, 2))
        
    except Exception as e:
        return render_template('result.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True, port=5000)
