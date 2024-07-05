import pandas as pd
import joblib

# Load the dataset
df = pd.read_excel(r"C:\Users\HP\Desktop\Fourth Year project\Code Deploymet\Childtraumadata.xlsx")

# Handle missing values
df.fillna(method='ffill', inplace=True)
df.fillna(method='bfill', inplace=True)

# Remove leading and trailing spaces from column names
df.columns = df.columns.str.strip()

# Remove leading and trailing spaces from string columns
df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)


df.head()

# Remove rows where trauma_stage is 'Ang'
df = df[df['trauma_stage'] != 'Ang']
#df = df[df['trauma_stage']]

# Verify the unique values in trauma_stage
print(df['trauma_stage'].unique())


# Inspect unique values to correct mappings
print(df['severity'].unique())
print(df['psychological_impact'].unique())
print(df['previous_trauma_history'].unique())
print(df['medical_history'].unique())
print(df['therapy_history'].unique())
print(df['lifestyle_factors'].unique())
print(df['resilience_factors'].unique())
print(df['exposure_to_stressors'].unique())
print(df['sleep_patterns'].unique())
print(df['emotional_regulation'].unique())

print(df['trauma_stage'].unique())


print(df.shape)

print(df['trauma_stage'].dtype)


print(df['trauma_stage'].isna().sum())


# Remove the unwanted 'trauma_stage' entry
df = df[df['trauma_stage'] != 'trauma_stage']


trauma_stage_forward_mapping = {
    'Anger': 0,
    'Sadness': 1,
    'Acceptance': 2,
    'Denial': 3,
    'Bargaining': 4,
    'Depression': 5
}


# Map the 'trauma_stage' column to numeric values
df['trauma_stage'] = df['trauma_stage'].map(trauma_stage_forward_mapping)


# Print unique values in trauma_stage column after conversion
print("Unique values in trauma_stage column after conversion:")
print(df['trauma_stage'].unique())


# Check for the number of NaN values in the 'trauma_stage' column
print(df['trauma_stage'].isna().sum())


# Print the unique text values that did not get converted to numeric values
unmapped_values = df[df['trauma_stage'].isna()]['trauma_stage']
print("Text values not mapped:")
print(unmapped_values.unique())


# Drop rows with NaN values in 'trauma_stage'
df = df.dropna(subset=['trauma_stage'])


df['trauma_stage'] = pd.to_numeric(df['trauma_stage'], errors='coerce')


# Define the mapping dictionaries
severity_map = {'Mild': 1, 'Moderate': 2, 'Severe': 3, 'Penetrating': 4, 'Not Classifiable': 5, 'Reserve': 6, 'Critical': 7}
psychological_impact_map = {'Low': 1, 'Medium': 2, 'High': 3, 'Yes': 4, 'No': 5, 'psychological_impact': 6}
previous_trauma_history_map = {'No': 1, 'Yes': 2, 'No previous trauma': 3, 'Witnessed a car accident': 4, 'None': 5,
                               'Witnessed domestic violence': 6, 'Car accident': 7, 'N/A': 8,
                               'Witnessed a violent altercation': 9, 'Witnessed a traumatic event': 10,
                               'Family history of addiction': 11, 'Yes (previous fall)': 12,
                               'Yes (previous fracture)': 13, 'previous_trauma_history': 14}
medical_history_map = {'None': 1, 'Asthma': 2, 'Depression': 3, 'No significant medical history': 4,
                      'History of asthma, Allergies': 5, 'ADHD, History of depression, Self-harm incidents': 6,
                      'Asthma, Previous fractures': 7, 'Born prematurely, Respiratory issues at birth': 8,
                      'History of allergies': 9, 'History of depression, Anxiety': 10,
                      'Previous fractures, History of asthma': 11, 'medical_history': 12, 'Hypertension': 13,
                      'Healthy': 14, 'No significant': 15, 'Allergies': 16, 'Diabetes': 17, 'Previous injury': 18}
therapy_history_map = {'None': 1, 'Counseling': 2, 'Individual and family therapy, Counseling': 3,
                      'Group therapy, Play therapy': 4,
                      'Cognitive-behavioral therapy, Dialectical behavior therapy': 5,
                      'Trauma-focused therapy, Eye movement desensitization and reprocessing': 6,
                      'Individual therapy, Art therapy': 7,
                      'Exposure therapy, Trauma-focused cognitive behavioral therapy': 8, 'N/A': 9,
                      'Occupational therapy, Sensory integration therapy': 10,
                      'Cognitive-behavioral therapy, Art therapy': 11,
                      'Dialectical behavior therapy, Group therapy': 12,
                      'Cognitive-behavioral therapy, Dialectical behavior': 13,
                      'Cognitive-behavioral therapy, Exposure therapy': 14,
                      'Cognitive-behavioral therapy': 15, 'therapy_history': 16, 'Physical Therapy': 17,
                      'Previous': 18, 'Ongoing': 19, 'Cognitive Behavioral': 20}
lifestyle_factors_map = {'Moderate exercise, No smoking': 1, 'No exercise, Occasional alcohol': 2,
                        'Active lifestyle, No alcohol': 3, 'Active lifestyle, No smoking': 4,
                        'No exercise, Smoking': 5, 'None': 6, 'Active play, No screen time': 7,
                        'No exercise, No smoking': 8, 'Regular exercise, Healthy diet': 9,
                        'Sedentary lifestyle, Fast food': 10, 'Active lifestyle, Balanced diet': 11,
                        'Outdoor activities, Balanced diet': 12, 'Regular sleep schedule': 13,
                        'Regular exercise': 14, 'Outdoor activities': 15, 'Active lifestyle': 16,
                        'Irregular sleep schedule': 17, 'Irregular': 18, 'Regular outdoor trips': 19,
                        'Playful Activities': 20, 'Structured Activities': 21, 'Occasional outdoor events': 22,
                        'Occasional travel': 23, 'Active social life': 24, 'Occasional home projects': 25,
                        'Frequent cooking': 26, 'Occasional shopping': 27, 'Occasional exercise': 28,
                        'Occasional attendance': 29, 'Occasional family events': 30, 'Occasional outdoor trips': 31,
                        'Occasional shopping trips': 32, 'Active, Outdoor Enthusiast': 33, 'Active, School Activities': 34,
                        'Active, Playful': 35, 'Athletic, Competitive': 36, 'Social, Active': 37,
                        'Active, Curious': 38, 'Playful, Active': 39, 'Active, Risk-taker': 40, 'Active, Social': 41,
                        'Sedentary, Office Worker': 42, 'Athletic, Health-conscious': 43, 'Social, Playful': 44,
                        'Regular cooking': 45, 'Occasional shopper': 46, 'Traveler': 47, 'Sedentary work': 48,
                        'Dance enthusiast': 49, 'Social events': 50, 'Religious activities': 51,
                        'Social activities': 52, 'Regular household': 53, 'Entertainment': 54, 'Office work': 55,
                        'Athletic': 56, 'Home improvement': 57, 'Adventure': 58, 'Shopping': 59, 'Cooking': 60,
                        'Traveling': 61, 'Art': 62, 'Balanced routine': 63, 'Healthy diet': 64, 'Balanced diet': 65,
                        'Structured routine': 66, 'Urban environment': 67, 'Coastal lifestyle': 68,
                        'Artistic pursuits': 69, 'Religious practices': 70, 'Beach lifestyle': 71, 'Active': 72,
                        'Sedentary': 73, 'Social': 74}
resilience_factors_map = {'High': 1, 'Moderate': 2, 'Low': 3, 'Positive mindset': 4, 'Social support': 5,
                         'Resilient mindset': 6, 'Supportive family': 7, 'None': 8, 'resilience_factors': 9,
                         'Optimism': 10, 'Social Support': 11, 'Resilience': 12, 'Adaptability': 13,
                         'Strong Family Ties': 14, 'Good Peer Relationships': 15, 'High Parental Support': 16,
                         'High Determination': 17, 'High Peer Support': 18, 'Good Work Relationships': 19,
                         'Emotional strength': 20}
exposure_to_stressors_map = {'High': 1, 'Moderate': 2, 'Low': 3, 'Minimal': 4, 'Occasional stress': 5,
                             'Moderate stress': 6, 'Minimal stress': 7, 'High stress': 8}
sleep_patterns_map = {'Poor': 1, 'Fair': 2, 'Good': 3, 'Excellent': 4, 'Regular': 5, 'Irregular': 6,
                      'Regular, disrupted': 7, 'Disrupted': 8, 'Disturbed': 9,
                      'Regular Sleep Pattern': 10, 'Occasional Night Waking': 11, 'Disrupted Sleep Pattern': 12}
emotional_regulation_map = {'Low': 1, 'Moderate': 2, 'High': 3, 'Effective': 4, 'Ineffective': 5,
                            'Stable': 6, 'Manageable': 7, 'Generally stable': 8, 'Unstable': 9,
                            'Good': 10, 'Adequate': 11, 'Inadequate': 12, 'Poor': 13}
# Print unique values in trauma_stage column
print("Unique values in trauma_stage column before conversion:")
print(df['trauma_stage'].unique())

# Convert trauma_stage column to numeric, handling errors
df['trauma_stage'] = pd.to_numeric(df['trauma_stage'], errors='coerce')

# Print unique values in trauma_stage column after conversion
print("Unique values in trauma_stage column after conversion:")
print(df['trauma_stage'].unique())

# Apply the mappings to the DataFrame
df['severity'] = df['severity'].map(severity_map)
df['psychological_impact'] = df['psychological_impact'].map(psychological_impact_map)
df['previous_trauma_history'] = df['previous_trauma_history'].map(previous_trauma_history_map)
df['medical_history'] = df['medical_history'].map(medical_history_map)
df['therapy_history'] = df['therapy_history'].map(therapy_history_map)
df['lifestyle_factors'] = df['lifestyle_factors'].map(lifestyle_factors_map)
df['resilience_factors'] = df['resilience_factors'].map(resilience_factors_map)
df['exposure_to_stressors'] = df['exposure_to_stressors'].map(exposure_to_stressors_map)
df['sleep_patterns'] = df['sleep_patterns'].map(sleep_patterns_map)
df['emotional_regulation'] = df['emotional_regulation'].map(emotional_regulation_map)
# Print DataFrame to verify changes
#print(df)






df.head()



# Define the features (X) and target variable (y)
X = df[['severity', 'psychological_impact', 'previous_trauma_history', 'medical_history', 'therapy_history', 
        'lifestyle_factors', 'resilience_factors', 'exposure_to_stressors', 'sleep_patterns', 'emotional_regulation']]
y = df['trauma_stage']  # Updated target variable to 'trauma_stage'


from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# Define the models to be used
# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
models = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'SVM': SVC(random_state=42)
}

# Define the models to be used
models = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'SVM': SVC(random_state=42)
}

# Define the hyperparameters for GridSearchCV
param_grid = {
    'Random Forest': {
        'n_estimators': [50, 100, 150],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    },
    'Logistic Regression': {
        'C': [0.1, 1, 10],
        'solver': ['liblinear', 'saga']
    },
    'SVM': {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf', 'poly']
    }
}

# Check for missing values in X_train and y_train
print("Missing values in X_train:")
print(X_train.isna().sum())

print("Missing values in y_train:")
print(y_train.isna().sum())


from sklearn.impute import SimpleImputer

# Define a simple imputer to fill missing values with the median (or mean, mode, etc.)
imputer = SimpleImputer(strategy='median')  # You can also use 'mean', 'most_frequent', etc.

# Fit the imputer on X_train and transform both X_train and X_test
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)


# Drop rows with missing values in X_train
X_train_clean = X_train.dropna()
y_train_clean = y_train[X_train_clean.index]

# Drop rows with missing values in X_test
X_test_clean = X_test.dropna()
y_test_clean = y_test[X_test_clean.index]


# Assuming you imputed missing values:
# Use X_train_imputed and X_test_imputed instead of X_train and X_test
for model_name, model in models.items():
    print(f"Training {model_name}...")
    grid_search = GridSearchCV(model, param_grid[model_name], cv=5, n_jobs=-1, verbose=1)
    grid_search.fit(X_train_imputed, y_train)  # Use the imputed data
    best_model = grid_search.best_estimator_

    # You can evaluate the best_model on X_test_imputed and y_test if needed
    print(f"Best parameters for {model_name}: {grid_search.best_params_}")
    print(f"Best score for {model_name}: {grid_search.best_score_}")


# Check for missing values in X_test
print("Missing values in X_test:")
print(X_test.isna().sum())


from sklearn.impute import SimpleImputer

# Define a simple imputer to fill missing values with the median
imputer = SimpleImputer(strategy='median')

# Fit the imputer on X_train and transform both X_train and X_test
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)



# Make predictions
y_pred = best_model.predict(X_test_imputed)

# Evaluate the model
print(f"Best parameters for {model_name}: {grid_search.best_params_}")
print(f"Accuracy score for {model_name}: {accuracy_score(y_test, y_pred)}")
print(f"Classification Report for {model_name}:\n{classification_report(y_test, y_pred)}")
print(f"Confusion Matrix for {model_name}:\n{confusion_matrix(y_test, y_pred)}")
print("\n" + "="*50 + "\n")

# Save the best model
joblib.dump(best_model, 'best_model.pkl')











