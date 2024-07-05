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

# Define trauma stages
trauma_stages = {
    0: {
        'stage': 'Anger',
        'characteristics': 'Characterized by frustration and anger.,Anxiety,Emotional exhaustion,High blood pressure,Insomnia,Passive-aggressive behavior,Alterations in thinking and mood,Continued obsession with the traumatic event,Depression,Difficulty concentrating,Intrusive memories,Muscle tension,Rapid breathing or hyperventilation',
        'solutions': ['Therapy', 'Anger management', 'Support groups']
    },
    1: {
        'stage': 'Sadness',
        'characteristics': 'Characterized by deep sadness and crying., exhaustion, confusion, sadness, anxiety, agitation, numbness, dissociation, confusion, physical arousal, and blunted affect.',
        'solutions': ['Counseling', 'Emotional support', 'Medication']
    },
    2: {
        'stage': 'Acceptance',
        'characteristics': 'Acceptance is the final stage of the trauma response cycle.',
        'solutions': ['Mindfulness', 'Support networks', 'Therapy']
    },
    3: {
        'stage': 'Denial',
        'characteristics': 'Characterized by refusal to acknowledge the trauma.,Many may isolate themselves from others while struggling in the denial stage',
        'solutions': ['Counseling', 'Education about trauma', 'Peer support']
    },
    4: {
        'stage': 'Bargaining',
        'characteristics': 'Characterized by attempts to negotiate out of trauma. The bargaining stage of trauma focuses on thoughts that take place within the mind. These thoughts occur as someone tries to explain how things could have been done differently or better. In a sense, these negotiations are an individuals thoughts attempting to exchange one thing for another',
        'solutions': ['Therapy', 'Support groups', 'Stress management']
    },
    5: {
        'stage': 'Depression',
        'characteristics': 'Characterized by feelings of severe despondency.negative emotions, guilt, shame, self-blame, social withdrawal, or social isolation.',
        'solutions': ['Counseling', 'Medication', 'Therapeutic activities']
    }
}

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
            # Handle other exceptions
            return jsonify({'error': str(e)}), 400
    return render_template('predict.html')

@app.route('/results')
def results():
    if 'username' not in session:
        return redirect(url_for('signup'))
    
    stage = request.args.get('stage', 'No result available')
    characteristics = request.args.get('characteristics', 'No characteristics available')
    solutions_str = request.args.get('solutions', '')
    solutions = solutions_str.split('|')
    characteristics_list = characteristics.split('|') if characteristics else []

    return render_template('results.html', stage=stage, characteristics=characteristics_list, solutions=solutions)

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('signup'))

@app.errorhandler(404)
def page_not_found(e):
    return jsonify({'error': 'Page not found'}), 404

if __name__ == '__main__':
    app.run(debug=True)
