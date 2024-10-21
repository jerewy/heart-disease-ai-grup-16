from flask import Flask, render_template, request, flash, redirect, url_for
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
import joblib
import numpy as np
import os

# Initialize the Flask app
app = Flask(__name__)

# Set up the database URI using an environment variable for deployment
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///predictions.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
migrate = Migrate(app, db)

app.secret_key = 'supersecretkey'  # Needed for flashing messages

# Load the model
try:
    model = joblib.load('calibrated_rf_model_isotonic_disease_model.pkl')
    print(f"Loaded model type: {type(model)}")
except Exception as e:
    model = None
    print(f"Error loading model: {e}")

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    age = db.Column(db.Integer)
    chol = db.Column(db.Integer)
    trestbps = db.Column(db.Integer)
    thalach = db.Column(db.Integer)
    oldpeak = db.Column(db.Float)
    ca = db.Column(db.Integer)
    cp = db.Column(db.Integer)
    fbs = db.Column(db.Integer)
    restecg = db.Column(db.Integer)
    exang = db.Column(db.Integer)
    slope = db.Column(db.Integer)
    thal = db.Column(db.Integer)
    result = db.Column(db.String)

    def __repr__(self):
        return f"<Prediction {self.id}: {self.result}>"

# Home route
@app.route('/')
def home():
    return render_template('dashboard.html')

# Dashboard route
@app.route('/dashboard')
def dashboard():
    total_predictions = Prediction.query.count()
    recent_predictions = Prediction.query.order_by(Prediction.id.desc()).limit(5).all()

    # New: Count risk levels
    low_risk_count = Prediction.query.filter(Prediction.result == 'Low Risk').count()
    moderate_risk_count = Prediction.query.filter(Prediction.result == 'Moderate Risk').count()
    high_risk_count = Prediction.query.filter(Prediction.result == 'High Risk').count()

    return render_template('dashboard.html', 
                           total_predictions=total_predictions, 
                           recent_predictions=recent_predictions, 
                           low_risk_count=low_risk_count,
                           moderate_risk_count=moderate_risk_count,
                           high_risk_count=high_risk_count)

# Prediction route
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            # Collect form inputs and handle conversion to appropriate types
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
        except ValueError as e:
            flash("Invalid input. Please check all fields and enter valid numbers.")
            return render_template('predict.html')

        # Validate input
        is_valid, error_message = validate_input(age, chol, trestbps, thalach, oldpeak, ca)
        if not is_valid:
            flash(error_message)
            return render_template('predict.html')

        # Prepare features for prediction as a DataFrame
        feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
        features = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]], columns=feature_names)
        
        if model:
            try:
                # Predict the probability using the model
                probabilities = model.predict_proba(features)
                heart_disease_risk = probabilities[0][1] * 100  # Convert to percentage
                
                # Classify risk level based on new thresholds
                if heart_disease_risk < 40:
                    risk_level = "Low Risk"
                elif 40 <= heart_disease_risk < 70:
                    risk_level = "Moderate Risk"
                else:
                    risk_level = "High Risk"

                # Save to the database
                new_prediction = Prediction(age=age, chol=chol, trestbps=trestbps,
                                            thalach=thalach, oldpeak=oldpeak,
                                            ca=ca, cp=cp, fbs=fbs, restecg=restecg,
                                            exang=exang, slope=slope, thal=thal,
                                            result=risk_level)
                db.session.add(new_prediction)
                db.session.commit()

                return render_template('result.html', result=f"{heart_disease_risk:.3f}", 
                                       risk_level=risk_level, prediction_id=new_prediction.id)

            except Exception as e:
                flash(f"Error making prediction: {e}")
                return render_template('predict.html')
        else:
            flash("Model is not available.")
            return render_template('predict.html')

    return render_template('predict.html')

# Route for history
@app.route('/history')
def history():
    predictions = Prediction.query.all()  # Get all predictions from the database
    return render_template('history.html', predictions=predictions)

# Delete prediction route
@app.route('/delete_prediction/<int:prediction_id>', methods=['POST'])
def delete_prediction(prediction_id):
    prediction_to_delete = Prediction.query.get(prediction_id)
    if prediction_to_delete:
        db.session.delete(prediction_to_delete)
        db.session.commit()
        flash("Prediction deleted successfully.")
    else:
        flash("Prediction not found.")
    return redirect(url_for('history'))

# Run the app
if __name__ == '__main__':
    with app.app_context():
        db.create_all()  # Creates the database tables based on your models
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
