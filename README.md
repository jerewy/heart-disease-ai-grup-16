# Heart Disease Prediction Web Application

This repository contains a web application for predicting heart disease using a machine learning model. The application is developed using Python with Flask as the web framework and leverages various libraries for data processing, visualization, and machine learning.

## Environment Setup

Follow these steps to set up and run the application:

### Prerequisites

- **Python 3.8 or higher** is required for this application. The minimum version is due to dependencies like Flask (3.0.3) and SQLAlchemy (2.0.36) that require Python 3.8+.
- It is recommended to use a virtual environment (e.g., `venv` or `conda`) to isolate dependencies.

### Installation Steps

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-repository-url.git
   cd your-repository-directory
2. **Create and Activate a Virtual Environment: Using virtualenv:**
   python -m venv venv
   source venv/bin/activate   # On Windows, use `venv\Scripts\activate`
   Or using conda:
   conda create -n heart-disease-env python=3.8
   conda activate heart-disease-env
3. **Install Required Dependencies: Use the provided requirements.txt to install all necessary libraries:**
   pip install -r requirements.txt
4. **Additional Setup: Ensure the machine learning model file**
   (calibrated_rf_model_isotonic_disease_model.pkl) is in the correct directory. This file is essential for generating predictions.

## Running the Application
1. **Start the Flask Server: Run the app.py file to launch the web server:**
   python app.py
2. **Access the Application:**
   - Open a browser and navigate to: http://127.0.0.1:5000
   - Alternatively, if on the same local network, access the app via the IP address shown in the terminal (e.g., http://192.168.1.105:5000).
4. **Use the Application:**
   Upload the required input data through the web interface.
   The application will process the data and provide predictions.

## Key Dependencies
The application uses the following major Python libraries:

- Flask 3.0.3: Web framework for building the application.
- Flask-SQLAlchemy 3.1.1: ORM for database integration.
- Matplotlib 3.8.4: Data visualization.
- Pandas 2.0.3: Data manipulation and analysis.
- Scikit-learn 1.2.2: Machine learning model training and predictions.
For a complete list of dependencies, refer to the requirements.txt file.
