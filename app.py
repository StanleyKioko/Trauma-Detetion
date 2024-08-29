from flask import Flask, request, jsonify, render_template, redirect, url_for, session
from flask_socketio import SocketIO, emit, join_room, leave_room
import joblib
import numpy as np
import sys
import pyrebase
import os
from datetime import datetime
from urllib.parse import quote as url_quote



from flask import Flask
app = Flask(__name__)
app.secret_key = os.urandom(24)
socketio = SocketIO(app, cors_allowed_origins="*")

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

# Load the saved model
model = joblib.load('best_model.pkl')
print("Model loaded successfully")

# Define trauma stages
trauma_stages = {
    0: {
        'stage': 'Anger',
        'characteristics': 'Outbursts of rage, frustration, irritability, and aggression. Difficulty controlling emotions and often blaming others.',
        'solutions': ['Therapy', 'Anger management', 'Support groups', 'Mindfulness techniques', 'Physical activities to channel energy']
    },
    1: {
        'stage': 'Sadness',
        'characteristics': 'Feelings of hopelessness, tearfulness, withdrawal from activities, and a sense of deep sorrow.',
        'solutions': ['Counseling', 'Emotional support', 'Medication', 'Engaging in hobbies', 'Connecting with loved ones']
    },
    2: {
        'stage': 'Acceptance',
        'characteristics': 'Acknowledgment of the trauma and its impact, a sense of calm, and readiness to move forward.',
        'solutions': ['Mindfulness', 'Support networks', 'Therapy', 'Developing new goals', 'Engaging in positive activities']
    },
    3: {
        'stage': 'Denial',
        'characteristics': 'Refusal to accept the reality of the trauma, avoidance of thoughts or discussions about it, and acting as if it didn’t happen.',
        'solutions': ['Counseling', 'Education about trauma', 'Peer support', 'Gradual exposure to trauma-related thoughts', 'Encouraging open communication']
    },
    4: {
        'stage': 'Bargaining',
        'characteristics': 'Attempting to make deals or promises to reverse or mitigate the trauma, often feeling guilt or seeking ways to regain control.',
        'solutions': ['Therapy', 'Support groups', 'Stress management', 'Journaling', 'Reflecting on personal strengths']
    },
    5: {
        'stage': 'Depression',
        'characteristics': 'Persistent sadness, lack of energy, loss of interest in activities, changes in appetite and sleep patterns, and feelings of worthlessness.',
        'solutions': ['Counseling', 'Medication', 'Therapeutic activities', 'Regular physical exercise', 'Building a daily routine']
    }
}


@app.route('/')
def index():
    if 'user' in session:
        role = session.get('role')
        if role == 'doctor':
            return redirect(url_for('doctor_dashboard'))
        elif role == 'patient':
            return redirect(url_for('patient_dashboard'))
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
        role = request.form['role']
        try:
            user = auth.create_user_with_email_and_password(email, password)
            user_id = user['localId']
            data = {
                "full_name": full_name,
                "email": email,
                "id_number": id_number,
                "country_code": country_code,
                "phone_number": phone_number,
                "role": role
            }
            db.child("users").child(user_id).set(data)
            session['user'] = user
            session['role'] = role
            if role == 'doctor':
                return redirect(url_for('doctor_dashboard'))
            else:
                return redirect(url_for('patient_dashboard'))
        except Exception as e:
            error_message = str(e)
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
            user_id = user['localId']
            user_data = db.child("users").child(user_id).get().val()
            session['role'] = user_data.get('role')
            if session['role'] == 'doctor':
                return redirect(url_for('doctor_dashboard'))
            elif session['role'] == 'patient':
                return redirect(url_for('patient_dashboard'))
        except Exception as e:
            error_message = str(e)
            return render_template('login.html', error=error_message)
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    session.pop('role', None)
    return redirect(url_for('login'))

@app.route('/doctor_dashboard', methods=['GET', 'POST'])
def doctor_dashboard():
    if 'user' not in session or session['role'] != 'doctor':
        return redirect(url_for('login'))
    
    user = session['user']
    user_id = user['localId']
    user_info = db.child("users").child(user_id).get().val()
    username = user_info.get('full_name', 'Unknown')
    
    try:
        predictions = db.child("predictions").order_by_child("doctor_id").equal_to(user_id).get()
        prediction_list = [pred.val() for pred in predictions.each()] if predictions.each() else []
    except Exception as e:
        print("Error fetching predictions:", e)
        prediction_list = []
    
    if request.method == 'POST':
        submission_id = request.form.get('submission_id')
        try:
            retrieved_prediction = db.child("predictions").child(submission_id).get().val()
            return render_template('doctor_dashboard.html', user=user, username=username, predictions=prediction_list, retrieved_prediction=retrieved_prediction)
        except Exception as e:
            print("Error retrieving prediction:", e)
            return render_template('doctor_dashboard.html', user=user, username=username, predictions=prediction_list, error=str(e))
    
    return render_template('doctor_dashboard.html', user=user, username=username, predictions=prediction_list)

@app.route('/patient_dashboard')
def patient_dashboard():
    if 'user' not in session or session['role'] != 'patient':
        return redirect(url_for('login'))
    
    user = session['user']
    user_id = user['localId']
    user_info = db.child("users").child(user_id).get().val()
    username = user_info.get('full_name', 'Unknown')
    
    # Retrieve doctor's ID for the form
    try:
        doctors = db.child("users").order_by_child("role").equal_to("doctor").get()
        doctor_list = [doc.key() for doc in doctors.each()] if doctors.each() else []
        doctor_id = doctor_list[0] if doctor_list else ''
    except Exception as e:
        print("Error fetching doctors:", e)
        doctor_id = ''
    
    return render_template('patient_dashboard.html', user=user, username=username, doctor_id=doctor_id)

@app.route('/submit_prediction', methods=['POST'])
def submit_prediction():
    if 'user' not in session or session['role'] != 'patient':
        return redirect(url_for('login'))
    
    user = session['user']
    user_id = user['localId']
    data = request.json
    
    try:
        features = np.array([
            float(data.get('severity', 0)),  # Default to 0 if not present
            float(data.get('psychological_impact', 0)),
            float(data.get('previous_trauma_history', 0)),
            float(data.get('medical_history', 0)),
            float(data.get('therapy_history', 0)),
            float(data.get('lifestyle_factors', 0)),
            float(data.get('resilience_factors', 0)),
            float(data.get('exposure_to_stressors', 0)),
            float(data.get('sleep_patterns', 0)),
            float(data.get('emotional_regulation', 0))
        ])
        
        prediction = model.predict([features])[0]
        trauma_stage = trauma_stages.get(prediction, {'stage': 'Unknown', 'characteristics': 'Unknown', 'solutions': ['Unknown']})
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        prediction_data = {
            "patient_id": user_id,
            "doctor_id": data.get('doctor_id', ''),
            "prediction": int(prediction),
            "stage": trauma_stage['stage'],
            "characteristics": trauma_stage['characteristics'],
            "solutions": trauma_stage['solutions'],
            "timestamp": timestamp
        }
        
        # Generate a unique ID for each prediction
        prediction_ref = db.child("predictions").push(prediction_data)
        prediction_id = prediction_ref['name']
        db.child("predictions").child(prediction_id).update({"prediction_id": prediction_id})
        
        return jsonify({"status": "success", "prediction_id": prediction_id})
    except Exception as e:
        print("Error in prediction submission:", e)
        return jsonify({"status": "error", "message": str(e)})

@socketio.on('message')
def handle_message(data):
    room = data.get('room')
    message = data.get('message')
    emit('message', {'message': message}, room=room)

@socketio.on('join')
def on_join(data):
    username = data.get('username')
    room = data.get('room')
    join_room(room)
    emit('message', {'message': f'{username} has entered the room.'}, room=room)

@socketio.on('leave')
def on_leave(data):
    username = data.get('username')
    room = data.get('room')
    leave_room(room)
    emit('message', {'message': f'{username} has left the room.'}, room=room)

if __name__ == '__main__':
    socketio.run(app, debug=True)