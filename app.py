from flask import Flask, render_template, request, flash
import joblib
import numpy as np

# Initialize the Flask app
app = Flask(__name__)
app.secret_key = 'supersecretkey'  # Needed for flashing messages

# Load the model
try:
    model = joblib.load('calibrated_rf_model_isotonic_disease_model.pkl')
    print(f"Loaded model type: {type(model)}")
except Exception as e:
    model = None
    print(f"Error loading model: {e}")

# Manually test the prediction with a sample input
if model is not None:
    test_features = np.array([[3, 0, 165, 1.0, 2, 55, 250, 140, 0, 2, 1, 1, 0]])  # Example values
    try:
        manual_prediction = model.predict(test_features)
        print(f"Manual Prediction Output: {manual_prediction}")
    except Exception as e:
        print(f"Error during manual prediction: {e}")
else:
    print("Model is not available.")


# Route for the home page (prediction form)
@app.route('/')
def home():
    return render_template('predict.html')

def validate_input(age, chol, trestbps, thalach, oldpeak, ca):
    if age < 0 or age > 120:
        return False, "Age must be between 0 and 120."
    if chol < 100 or chol > 600:
        return False, "Cholesterol must be between 100 and 600 mg/dL."
    if trestbps < 80 or trestbps > 200:
        return False, "Resting Blood Pressure must be between 80 and 200 mm Hg."
    if thalach < 60 or thalach > 200:
        return False, "Max Heart Rate must be between 60 and 200 beats per minute."
    if oldpeak < 0 or oldpeak > 6:
        return False, "ST Depression must be between 0 and 6."
    if ca < 0 or ca > 3:
        return False, "Number of Major Vessels must be between 0 and 3."
    return True, None


# Route to handle the form submission and make a prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if request.method == 'POST':
            try:
                # Collect form inputs
                cp = int(request.form.get('cp', None))
                ca = int(request.form.get('ca', None))
                thalach = int(request.form.get('thalach', None))
                oldpeak = float(request.form.get('oldpeak', None))
                thal = int(request.form.get('thal', None))
                age = int(request.form.get('age', None))
                chol = int(request.form.get('chol', None))
                trestbps = int(request.form.get('trestbps', None))
                exang = int(request.form.get('exang', None))
                slope = int(request.form.get('slope', None))
                sex = int(request.form.get('sex', None))
                restecg = int(request.form.get('restecg', None))
                fbs = int(request.form.get('fbs', None))
            except ValueError:
                flash("Invalid input. Please check all fields and enter valid numbers.")
                return render_template('predict.html')

            # Validate input for age, cholesterol, trestbps, etc.
            is_valid, error_message = validate_input(age, chol, trestbps, thalach, oldpeak, ca)
            if not is_valid:
                flash(error_message)
                return render_template('predict.html')


            # All fields are now valid, proceed with the prediction
            features = np.array([[cp, ca, thalach, oldpeak, thal, age, chol, trestbps, exang, slope, sex, restecg, fbs]])
            print(f"Feature shape: {features.shape}")

            if model:
                try:
                    # Predict the probability using the model
                    probabilities = model.predict_proba(features)
                    print(f"Probabilities: {probabilities}")

                    # Extract the probability of heart disease (assuming '1' represents heart disease)
                    heart_disease_risk = probabilities[0][1] * 100  # Convert to percentage
                    print(f"Heart disease risk: {heart_disease_risk}%")

                    result = f"{heart_disease_risk:.3f}"

                except Exception as e:
                    flash(f"Error making prediction: {e}")
                    print(f"Prediction error: {e}")
                    result = None
            else:
                flash("Model is not available.")
                result = None

            return render_template('result.html', result=result)

    except Exception as e:
        flash(f"An unexpected error occurred: {e}")
        print(f"Error: {e}")
        return render_template('predict.html', result=None)


if __name__ == '__main__':
    app.run(debug=True)
