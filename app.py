from flask import Flask, request, jsonify, render_template, redirect, url_for, session
import joblib
import numpy as np
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Simulated user database
users = {}

# Load the saved model
model = joblib.load('best_model.pkl')
print("Model loaded successfully")

@app.route('/')
def index():
    if 'username' in session:
        return redirect(url_for('home'))
    return redirect(url_for('signup'))

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        if email in users:
            return jsonify({'error': 'User already exists'}), 400
        users[email] = {
            'username': username,
            'email': email,
            'password': generate_password_hash(password)
        }
        session['username'] = username
        return redirect(url_for('home'))
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = users.get(email)
        if user and check_password_hash(user['password'], password):
            session['username'] = user['username']
            return redirect(url_for('home'))
        return jsonify({'error': 'Invalid credentials'}), 400
    return render_template('signup.html')

@app.route('/home')
def home():
    if 'username' not in session:
        return redirect(url_for('signup'))
    return render_template('home.html', username=session['username'])

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if 'username' not in session:
        return redirect(url_for('signup'))
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
            
            # Convert features to a numpy array and reshape for prediction
            features = np.array(features).reshape(1, -1)
            
            # Make the prediction
            prediction = model.predict(features)

            # Define trauma stages
            trauma_stage = {
                0: 'Anger',
                1: 'Sadness',
                2: 'Acceptance',
                3: 'Denial',
                4: 'Bargaining',
                5: 'Depression'
            }
            predicted_stage = trauma_stage.get(prediction[0], 'Unknown')  # Default to 'Unknown' if prediction is invalid
            
            # Redirect to the results page with the prediction result
            return redirect(url_for('results', result=predicted_stage))
        
        except KeyError as e:
            # Handle missing data keys
            return jsonify({'error': f'Missing key: {str(e)}'}), 400
        except Exception as e:
            # Handle other exceptions
            return jsonify({'error': str(e)}), 400
    return render_template('predict.html')

@app.route('/results')
def results():
    if 'username' not in session:
        return redirect(url_for('signup'))
    result = request.args.get('result', 'No result available')
    return render_template('results.html', result=result)

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('signup'))

@app.errorhandler(404)
def page_not_found(e):
    return jsonify({'error': 'Page not found'}), 404

if __name__ == '__main__':
    app.run(debug=True)
