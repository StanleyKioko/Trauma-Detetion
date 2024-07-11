from flask import Flask, request, jsonify, render_template, redirect, url_for, session
import joblib
import numpy as np
from werkzeug.security import generate_password_hash, check_password_hash
import pyrebase
import os

app = Flask(__name__)

# Firebase Configuration
firebaseConfig = {
    'apiKey': "AIzaSyCJ_3tFr0xvhVRtnwPG2SpL0FfZyUP9-V4",
    'authDomain': "database-22a18.firebaseapp.com",
    'databaseURL': "https://database-22a18-default-rtdb.firebaseio.com",
    'projectId': "database-22a18",
    'storageBucket': "database-22a18.appspot.com",
    'messagingSenderId': "592734301931",
    'appId': "1:592734301931:web:283a373c93106c891218ae"
}

# Initialize Firebase
firebase = pyrebase.initialize_app(firebaseConfig)
auth = firebase.auth()
db = firebase.database()

# Set a secret key for session management
app.secret_key = os.urandom(24)

# Load the saved model
model = joblib.load('best_model.pkl')
print("Model loaded successfully")

# Define trauma stages
trauma_stages = {
    0: {
        'stage': 'Anger',
        'characteristics': 'Characterized by frustration and anger, Anxiety, Emotional exhaustion, High blood pressure, Insomnia, Passive-aggressive behavior, Alterations in thinking and mood, Continued obsession with the traumatic event, Depression, Difficulty concentrating, Intrusive memories, Muscle tension, Rapid breathing or hyperventilation',
        'solutions': ['Therapy', 'Anger management', 'Support groups']
    },
    1: {
        'stage': 'Sadness',
        'characteristics': 'Characterized by deep sadness and crying, exhaustion, confusion, sadness, anxiety, agitation, numbness, dissociation, confusion, physical arousal, and blunted affect.',
        'solutions': ['Counseling', 'Emotional support', 'Medication']
    },
    2: {
        'stage': 'Acceptance',
        'characteristics': 'Acceptance is the final stage of the trauma response cycle.',
        'solutions': ['Mindfulness', 'Support networks', 'Therapy']
    },
    3: {
        'stage': 'Denial',
        'characteristics': 'Characterized by refusal to acknowledge the trauma. Many may isolate themselves from others while struggling in the denial stage.',
        'solutions': ['Counseling', 'Education about trauma', 'Peer support']
    },
    4: {
        'stage': 'Bargaining',
        'characteristics': 'Characterized by attempts to negotiate out of trauma. The bargaining stage of trauma focuses on thoughts that take place within the mind. These thoughts occur as someone tries to explain how things could have been done differently or better.',
        'solutions': ['Therapy', 'Support groups', 'Stress management']
    },
    5: {
        'stage': 'Depression',
        'characteristics': 'Characterized by feelings of severe despondency, negative emotions, guilt, shame, self-blame, social withdrawal, or social isolation.',
        'solutions': ['Counseling', 'Medication', 'Therapeutic activities']
    }
}

@app.route('/')
def index():
    if 'user' in session:
        return redirect(url_for('home'))
    return redirect(url_for('signup'))

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        full_name = request.form['username']
        id_number = request.form['id_number']
        country_code = request.form['country_code']
        phone_number = request.form['phone_number']
        try:
            user = auth.create_user_with_email_and_password(email, password)
            user_id = user['localId']
            data = {
                "full_name": full_name,
                "email": email,
                "id_number": id_number,
                "country_code": country_code,
                "phone_number": phone_number
            }
            db.child("users").child(user_id).set(data)
            session['user'] = user
            return redirect(url_for('home'))
        except Exception as e:
            error_message = str(e)
            if "EMAIL_EXISTS" in error_message:
                error_message = "The email address is already in use. Please use a different email address."
            elif "INVALID_EMAIL" in error_message:
                error_message = "Invalid email address. Please check your email and try again."
            elif "WEAK_PASSWORD" in error_message:
                error_message = "The password is too weak. Please choose a stronger password."
            elif "USER_DISABLED" in error_message:
                error_message = "This user account has been disabled."
            elif "INVALID_PASSWORD" in error_message:
                error_message = "The password is invalid or the user does not have a password."
            elif "USER_NOT_FOUND" in error_message:
                error_message = "There is no user corresponding to this email address."
            elif "OPERATION_NOT_ALLOWED" in error_message:
                error_message = "Operation not allowed. Please contact support."
            else:
                error_message = "An unexpected error occurred: " + error_message
            return render_template('signup.html', error=error_message)
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        try:
            user = auth.sign_in_with_email_and_password(email, password)
            session['user'] = user
            return redirect(url_for('home'))
        except Exception as e:
            error_message = str(e)
            if "INVALID_EMAIL" in error_message:
                error_message = "Invalid email address. Please check your email and try again."
            elif "USER_NOT_FOUND" in error_message:
                error_message = "No account found with this email address. Please sign up."
            elif "INVALID_PASSWORD" in error_message:
                error_message = "Incorrect password. Please try again."
            elif "USER_DISABLED" in error_message:
                error_message = "This user account has been disabled."
            else:
                error_message = "An unexpected error occurred: " + error_message
            return render_template('login.html', error=error_message)
    return render_template('login.html')

@app.route('/home')
def home():
    if 'user' not in session:
        return redirect(url_for('login'))
    user = session['user']
    user_info = db.child("users").child(user['localId']).get().val()
    username = user_info['full_name']
    return render_template('home.html', username=username)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if 'user' not in session:
        return redirect(url_for('login'))
    if request.method == 'POST':
        try:
            data = request.form  # Get the form data

            # Extract the features from the form data
            features = [
                float(data['severity']),
                float(data['psychological_impact']),
                float(data['previous_trauma_history']),
                float(data['medical_history']),
                float(data['therapy_history']),
                float(data['lifestyle_factors']),
                float(data['resilience_factors']),
                float(data['exposure_to_stressors']),
                float(data['sleep_patterns']),
                float(data['emotional_regulation'])
            ]

            # Check if all features are zero
            if all(feature == 0 for feature in features):
                return redirect(url_for('results', stage='Unknown', characteristics='', solutions=''))
            
            # Convert features to a numpy array and reshape for prediction
            features = np.array(features).reshape(1, -1)
            
            # Make the prediction
            prediction = model.predict(features)

            # Get the predicted trauma stage
            predicted_stage = trauma_stages.get(prediction[0], None)

            if predicted_stage is None:
                return jsonify({'error': 'Invalid prediction'}), 400

            # Prepare the characteristics and solutions for URL
            characteristics = '|'.join(predicted_stage['characteristics'].split(','))
            solutions = '|'.join(predicted_stage['solutions'])

            # Redirect to the results page with the prediction result
            return redirect(url_for('results', 
                                    stage=predicted_stage['stage'], 
                                    characteristics=characteristics, 
                                    solutions=solutions))
        
        except KeyError as e:
            # Handle missing data keys
            return jsonify({'error': f'Missing key: {str(e)}'}), 400
        except Exception as e:
            # Handle any other exceptions
            return jsonify({'error': f'An error occurred: {str(e)}'}), 500

    return render_template('predict.html')

@app.route('/results')
def results():
    if 'user' not in session:
        return redirect(url_for('login'))
    stage = request.args.get('stage', 'Unknown')
    characteristics = request.args.get('characteristics', '')
    solutions = request.args.get('solutions', '')
    characteristics = characteristics.replace('|', ', ')
    solutions = solutions.replace('|', ', ')
    return render_template('results.html', stage=stage, characteristics=characteristics, solutions=solutions)

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

@app.route('/make_another_prediction')
def make_another_prediction():
    return redirect(url_for('predict'))

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

if __name__ == '__main__':
    app.run(debug=True)
