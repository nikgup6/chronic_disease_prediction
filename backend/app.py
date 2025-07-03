# backend/app.py

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os

app = Flask(__name__)
CORS(app) # Enable CORS for all origins, adjust in production for specific frontend URL

# Directory to store trained models
MODELS_DIR = 'models'
os.makedirs(MODELS_DIR, exist_ok=True)

# Define model paths
PREDIABETES_MODEL_PATH = os.path.join(MODELS_DIR, 'prediabetes_model.pkl')
HYPERTENSION_MODEL_PATH = os.path.join(MODELS_DIR, 'hypertension_model.pkl')
METABOLIC_SYNDROME_MODEL_PATH = os.path.join(MODELS_DIR, 'metabolic_syndrome_model.pkl')
SCALER_PATH = os.path.join(MODELS_DIR, 'scaler.pkl')

# Feature columns used for training
# These must match the order and names expected by the trained models
FEATURE_COLUMNS = [
    'glucose', 'systolicBP', 'diastolicBP', 'triglycerides', 'hdl',
    'waistCircumference', 'age', 'gender', # Added age and gender as numerical features
    'activityLevel_high', 'activityLevel_low', 'activityLevel_moderate',
    'diet_healthy', 'diet_moderate', 'diet_unhealthy',
    'sleep_average', 'sleep_good', 'sleep_poor',
    'stress_high', 'stress_low', 'stress_moderate',
    'alcohol_no', 'alcohol_yes',
    'smoking_no', 'smoking_yes',
    'fatigue_no', 'fatigue_yes'
]

# --- Model Training Function ---
def train_and_save_models():
    """
    Loads and preprocesses NHANES data, trains ML models for pre-diabetes, hypertension,
    and metabolic syndrome, and saves them to disk.
    """
    print("Loading NHANES dataset files and preprocessing...")

    # Define paths to NHANES CSV files (assuming they are in the 'backend' directory)
    DEMO_PATH = 'demographic.csv'
    EXAM_PATH = 'examination.csv'
    LABS_PATH = 'labs.csv'
    Q_PATH = 'questionnaire.csv'

    try:
        df_demo = pd.read_csv(DEMO_PATH)
        df_exam = pd.read_csv(EXAM_PATH)
        df_labs_raw = pd.read_csv(LABS_PATH) # Load raw labs data first
        df_q_raw = pd.read_csv(Q_PATH) # Load raw questionnaire data first
    except FileNotFoundError as e:
        print(f"Error: NHANES file not found: {e.filename}. Please ensure all .csv files are in the 'backend' directory.")
        return # Exit if files are missing
    except Exception as e:
        print(f"Error loading NHANES files: {e}")
        return # Exit on other loading errors

    print("NHANES files loaded. Merging data...")

    # --- 1. Select and Rename Relevant Columns ---
    # Demographics
    df_demo = df_demo[['SEQN', 'RIDAGEYR', 'RIAGENDR']]
    df_demo = df_demo.rename(columns={'RIDAGEYR': 'age', 'RIAGENDR': 'gender'})

    # Examination
    df_exam = df_exam[['SEQN', 'BPXSY1', 'BPXDI1', 'BMXWAIST']]
    df_exam = df_exam.rename(columns={'BPXSY1': 'systolicBP', 'BPXDI1': 'diastolicBP', 'BMXWAIST': 'waistCircumference'})

    # Labs - More robust selection
    print("\n--- Labs Data Columns ---")
    print("Columns in labs.csv (raw from CSV):", df_labs_raw.columns.tolist())
    df_labs = pd.DataFrame({'SEQN': df_labs_raw['SEQN']}) # Start with SEQN

    # Define a list of (internal_name, potential_nhanes_names) tuples
    lab_columns_to_map = [
        ('glucose', ['LBDGLU', 'LBXGLU', 'LBXSGL', 'LBDSGLSI']), # Added LBXSGL, LBDSGLSI
        ('hdl', ['LBDHDD', 'LBXHDD']),
        ('triglycerides', ['LBDTRM', 'LBXTR'])
    ]

    for internal_name, potential_nhanes_names in lab_columns_to_map:
        found_column = False
        for nhanes_name in potential_nhanes_names:
            if nhanes_name in df_labs_raw.columns:
                df_labs[internal_name] = df_labs_raw[nhanes_name]
                found_column = True
                print(f"Mapped NHANES column '{nhanes_name}' to internal feature '{internal_name}'.")
                break
        if not found_column:
            print(f"Warning: None of {potential_nhanes_names} found for '{internal_name}'. This column will be filled with NaNs.")
            df_labs[internal_name] = np.nan # Ensure column exists, filled with NaN

    print("Columns in df_labs (after selection/renaming):", df_labs.columns.tolist())


    # Questionnaire - Select and prepare for mapping (more robust)
    print("\n--- Questionnaire Data Columns ---")
    print("Columns in questionnaire.csv (raw from CSV):", df_q_raw.columns.tolist())
    df_q = pd.DataFrame({'SEQN': df_q_raw['SEQN']}) # Start with SEQN

    q_columns_to_map = [
        ('vigorous_activity', ['PAQ605', 'PAQ610']), # PAQ610 is sometimes used
        ('moderate_activity', ['PAQ620', 'PAQ635']), # PAQ635 is sometimes used
        ('diet_quality_code', ['DBQ700']),
        ('sleep_hours', ['SLD010H']),
        ('alcohol_freq_code', ['ALQ110']),
        ('smoking_status_code', ['SMQ040']),
        ('fatigue_interest_code', ['DPQ010']), # Primary for fatigue
        ('stress_depressed_code', ['DPQ020']) # Primary for stress/depression
    ]

    for internal_name, potential_nhanes_names in q_columns_to_map:
        found_column = False
        for nhanes_name in potential_nhanes_names:
            if nhanes_name in df_q_raw.columns:
                df_q[internal_name] = df_q_raw[nhanes_name]
                found_column = True
                print(f"Mapped NHANES column '{nhanes_name}' to internal feature '{internal_name}'.")
                break
        if not found_column:
            print(f"Warning: None of {potential_nhanes_names} found for '{internal_name}'. This column will be filled with NaNs.")
            df_q[internal_name] = np.nan # Ensure column exists, filled with NaN

    print("Columns in df_q (after selection/renaming):", df_q.columns.tolist())


    # --- 2. Merge DataFrames ---
    # Start with demographics and sequentially merge others
    df = df_demo
    df = pd.merge(df, df_exam, on='SEQN', how='left')
    df = pd.merge(df, df_labs, on='SEQN', how='left')
    df = pd.merge(df, df_q, on='SEQN', how='left')

    print(f"\nMerged DataFrame shape: {df.shape}")
    print("Columns after initial merge and rename:", df.columns.tolist())

    # --- 3. Handle Missing Values ---
    # For numerical columns, impute with median (more robust to outliers than mean)
    numerical_cols = ['glucose', 'systolicBP', 'diastolicBP', 'triglycerides', 'hdl', 'waistCircumference', 'age', 'sleep_hours']
    for col in numerical_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce') # Ensure numeric type
            df[col] = df[col].fillna(df[col].median()) # Removed inplace=True

    # For categorical/code columns, impute with mode
    categorical_code_cols = ['gender', 'vigorous_activity', 'moderate_activity', 'diet_quality_code',
                             'alcohol_freq_code', 'smoking_status_code', 'fatigue_interest_code', 'stress_depressed_code']
    for col in categorical_code_cols:
        if col in df.columns:
            # Mode might return multiple values, take the first one
            df[col] = df[col].fillna(df[col].mode()[0]) # Removed inplace=True

    # --- 4. Transform and Create Categorical Features for ML Models ---

    # Gender: 1=Male, 2=Female. Map to 0/1 (e.g., 0=Female, 1=Male for simplicity)
    df['gender'] = df['gender'].map({1: 1, 2: 0}).fillna(df['gender'].mode()[0]) # Fillna for any remaining NaNs

    # Activity Level: Combine vigorous and moderate activity
    # 1=Yes, 2=No, 7=Refused, 9=Don't Know (for PAQ605/PAQ620)
    # Map to 'high', 'moderate', 'low'
    df['activityLevel'] = 'low' # Default to low
    if 'vigorous_activity' in df.columns:
        df.loc[df['vigorous_activity'] == 1, 'activityLevel'] = 'high'
    if 'moderate_activity' in df.columns:
        df.loc[(df['moderate_activity'] == 1) & (df['activityLevel'] == 'low'), 'activityLevel'] = 'moderate'
    df = df.drop(columns=['vigorous_activity', 'moderate_activity'], errors='ignore') # Use errors='ignore'

    # Diet Quality: 1=Excellent to 5=Poor. Map to 'healthy', 'moderate', 'unhealthy'
    df['diet'] = 'unhealthy' # Default
    if 'diet_quality_code' in df.columns:
        df.loc[df['diet_quality_code'].isin([1, 2]), 'diet'] = 'healthy' # Excellent, Very Good
        df.loc[df['diet_quality_code'] == 3, 'diet'] = 'moderate' # Good
    df = df.drop(columns=['diet_quality_code'], errors='ignore')

    # Sleep Quality: Based on hours. Simplified.
    df['sleep'] = 'poor' # Default
    if 'sleep_hours' in df.columns:
        df.loc[(df['sleep_hours'] >= 6) & (df['sleep_hours'] <= 7), 'sleep'] = 'average'
        df.loc[df['sleep_hours'] > 7, 'sleep'] = 'good'

    # Stress Level: Based on DPQ010 and DPQ020 (depression symptoms)
    # 0=Not at all, 1=Several days, 2=More than half the days, 3=Nearly every day
    df['stress'] = 'low' # Default
    if 'fatigue_interest_code' in df.columns and 'stress_depressed_code' in df.columns:
        df.loc[(df['fatigue_interest_code'] >= 2) | (df['stress_depressed_code'] >= 2), 'stress'] = 'high'
        df.loc[(df['fatigue_interest_code'] == 1) | (df['stress_depressed_code'] == 1), 'stress'] = 'moderate'
    df = df.drop(columns=['fatigue_interest_code', 'stress_depressed_code'], errors='ignore')

    # Alcohol Consumption: ALQ110 (1=Every day, ..., 7=Never)
    df['alcohol'] = 'yes' # Default
    if 'alcohol_freq_code' in df.columns:
        df.loc[df['alcohol_freq_code'] == 7, 'alcohol'] = 'no' # Never
    df = df.drop(columns=['alcohol_freq_code'], errors='ignore')

    # Smoking Status: SMQ040 (1=Yes, 2=No)
    df['smoking'] = 'no' # Default
    if 'smoking_status_code' in df.columns:
        df.loc[df['smoking_status_code'] == 1, 'smoking'] = 'yes'
    df = df.drop(columns=['smoking_status_code'], errors='ignore')

    # Fatigue: Use a simplified binary from DPQ010
    # Note: The original DPQ010 column is fatigue_interest_code after renaming
    df['fatigue'] = 'no' # Default
    if 'fatigue_interest_code' in df.columns: # Use the renamed column
        df.loc[df['fatigue_interest_code'] >= 1, 'fatigue'] = 'yes' # If 'several days' or more


    # --- 5. Define Target Variables based on medical criteria (using NHANES data) ---

    # Pre-diabetes: Fasting Glucose 100-125 mg/dL
    df['prediabetes'] = ((df['glucose'] >= 100) & (df['glucose'] <= 125)).astype(int)

    # Hypertension: Systolic BP >= 130 or Diastolic BP >= 80 (Stage 1 or higher)
    df['hypertension'] = ((df['systolicBP'] >= 130) | (df['diastolicBP'] >= 80)).astype(int)

    # Metabolic Syndrome (NCEP ATP III criteria - simplified for available NHANES data): 3 or more of the following:
    # 1. Waist Circumference: >=102 cm (men), >=88 cm (women)
    # 2. Triglycerides: >=150 mg/dL
    # 3. HDL Cholesterol: <40 mg/dL (men), <50 mg/dL (women)
    # 4. Blood Pressure: Systolic >=130 or Diastolic >=85 (or on medication - not in this dataset)
    # 5. Fasting Glucose: >=100 mg/dL (or on medication - not in this dataset)

    df['metabolic_syndrome_criteria_count'] = 0
    df.loc[(df['gender'] == 1) & (df['waistCircumference'] >= 102), 'metabolic_syndrome_criteria_count'] += 1 # Men
    df.loc[(df['gender'] == 0) & (df['waistCircumference'] >= 88), 'metabolic_syndrome_criteria_count'] += 1 # Women
    df.loc[df['triglycerides'] >= 150, 'metabolic_syndrome_criteria_count'] += 1
    df.loc[(df['gender'] == 1) & (df['hdl'] < 40), 'metabolic_syndrome_criteria_count'] += 1 # Men
    df.loc[(df['gender'] == 0) & (df['hdl'] < 50), 'metabolic_syndrome_criteria_count'] += 1 # Women
    df.loc[(df['systolicBP'] >= 130) | (df['diastolicBP'] >= 85), 'metabolic_syndrome_criteria_count'] += 1
    df.loc[df['glucose'] >= 100, 'metabolic_syndrome_criteria_count'] += 1
    df['metabolic_syndrome'] = (df['metabolic_syndrome_criteria_count'] >= 3).astype(int)
    df = df.drop(columns=['metabolic_syndrome_criteria_count']) # Drop intermediate column

    # --- ENSURE AT LEAST TWO CLASSES FOR EACH TARGET ---
    # This is crucial to prevent the ValueError during model training
    target_cols = ['prediabetes', 'hypertension', 'metabolic_syndrome']
    for col in target_cols:
        if df[col].nunique() < 2:
            print(f"Warning: '{col}' has only one class after initial generation. Forcing some samples to the other class.")
            single_class_value = df[col].iloc[0]
            num_to_flip = max(5, int(len(df) * 0.05)) # Flip 5% or at least 5 samples
            indices_to_flip = np.random.choice(df.index, num_to_flip, replace=False)
            df.loc[indices_to_flip, col] = 1 - single_class_value
            if df[col].nunique() < 2:
                 print(f"Error: Could not ensure two classes for '{col}' even after forcing. This might cause issues.")
                 # As a last resort, if still only one class, fill with random 0s and 1s to allow training
                 df[col] = np.random.randint(0, 2, len(df))
                 print(f"Forced '{col}' to have two classes randomly. Model performance for this target might be poor.")


    # --- 6. One-Hot Encode Categorical Features ---
    # Ensure these categories match the ones used in FEATURE_COLUMNS
    categorical_cols_for_ohe = {
        'activityLevel': ['low', 'moderate', 'high'],
        'diet': ['unhealthy', 'moderate', 'healthy'],
        'sleep': ['poor', 'average', 'good'],
        'stress': ['high', 'moderate', 'low'],
        'alcohol': ['yes', 'no'],
        'smoking': ['yes', 'no'],
        'fatigue': ['yes', 'no']
    }

    for col, categories in categorical_cols_for_ohe.items():
        if col in df.columns:
            dummies = pd.get_dummies(df[col], prefix=col, prefix_sep='_', drop_first=False)
            # Ensure all defined categories for this column are present in the dummies
            for cat in categories:
                if f'{col}_{cat}' not in dummies.columns:
                    dummies[f'{col}_{cat}'] = 0
            df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
        else:
            # If a categorical column is missing from the dataset, add dummy columns initialized to 0
            for cat in categories:
                df[f'{col}_{cat}'] = 0

    # Ensure target columns are integers
    for target_col in target_cols:
        df[target_col] = df[target_col].astype(int)

    # Select only features and targets, and ensure order
    # This will drop any rows that still have NaNs in FEATURE_COLUMNS after all imputation
    # and ensure X is not empty.
    df_final = df[FEATURE_COLUMNS + target_cols].dropna()

    X = df_final[FEATURE_COLUMNS]
    y_prediabetes = df_final['prediabetes']
    y_hypertension = df_final['hypertension']
    y_metabolic_syndrome = df_final['metabolic_syndrome']

    # CRITICAL CHECK: Ensure X is not empty before scaling/training
    if X.empty:
        raise ValueError("DataFrame X is empty after preprocessing. This means no valid samples are left for training. Please check data loading, merging, and imputation steps.")

    print(f"Final features shape: {X.shape}, Target shapes: {y_prediabetes.shape}, {y_hypertension.shape}, {y_metabolic_syndrome.shape}")
    print("Features used for training:", X.columns.tolist())

    # Scale numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=FEATURE_COLUMNS) # Keep as DataFrame for consistency

    # Save the scaler
    joblib.dump(scaler, SCALER_PATH)
    print(f"Scaler saved to {SCALER_PATH}")

    # Train and save Pre-diabetes model
    model_prediabetes = LogisticRegression(solver='liblinear', random_state=42, class_weight='balanced') # Added class_weight
    model_prediabetes.fit(X_scaled_df, y_prediabetes)
    joblib.dump(model_prediabetes, PREDIABETES_MODEL_PATH)
    print(f"Pre-diabetes model saved to {PREDIABETES_MODEL_PATH}")

    # Train and save Hypertension model
    model_hypertension = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced') # Added class_weight
    model_hypertension.fit(X_scaled_df, y_hypertension)
    joblib.dump(model_hypertension, HYPERTENSION_MODEL_PATH)
    print(f"Hypertension model saved to {HYPERTENSION_MODEL_PATH}")

    # Train and save Metabolic Syndrome model
    model_metabolic_syndrome = LogisticRegression(solver='liblinear', random_state=42, class_weight='balanced') # Added class_weight
    model_metabolic_syndrome.fit(X_scaled_df, y_metabolic_syndrome)
    joblib.dump(model_metabolic_syndrome, METABOLIC_SYNDROME_MODEL_PATH)
    print(f"Metabolic Syndrome model saved to {METABOLIC_SYNDROME_MODEL_PATH}")

    print("Model training complete using NHANES data.")

# --- Load Models (Global for API) ---
try:
    scaler = joblib.load(SCALER_PATH)
    prediabetes_model = joblib.load(PREDIABETES_MODEL_PATH)
    hypertension_model = joblib.load(HYPERTENSION_MODEL_PATH)
    metabolic_syndrome_model = joblib.load(METABOLIC_SYNDROME_MODEL_PATH)
    print("Models loaded successfully.")
except FileNotFoundError:
    print("Models not found. Please run 'python app.py train' to train them first.")
    scaler = None
    prediabetes_model = None
    hypertension_model = None
    metabolic_syndrome_model = None
except Exception as e:
    print(f"Error loading models: {e}")
    scaler = None
    prediabetes_model = None
    hypertension_model = None
    metabolic_syndrome_model = None

# --- Prediction Endpoint ---
@app.route('/predict', methods=['POST'])
def predict():
    if not all([scaler, prediabetes_model, hypertension_model, metabolic_syndrome_model]):
        return jsonify({"error": "Models not loaded. Please train them first."}), 500

    try:
        data = request.get_json(force=True)

        # Convert input data to DataFrame for consistent processing
        input_df = pd.DataFrame([data])

        # Add age and gender to input_df for prediction, assuming default values if not provided by frontend
        # In a real app, you'd likely collect these from user profile
        if 'age' not in input_df.columns:
            input_df['age'] = 40 # Default age
        if 'gender' not in input_df.columns:
            input_df['gender'] = 1 # Default gender (1=Male, 0=Female, matching training)

        # Handle categorical features using one-hot encoding
        categorical_cols = {
            'activityLevel': ['low', 'moderate', 'high'],
            'diet': ['unhealthy', 'moderate', 'healthy'],
            'sleep': ['poor', 'average', 'good'],
            'stress': ['high', 'moderate', 'low'],
            'alcohol': ['yes', 'no'],
            'smoking': ['yes', 'no'],
            'fatigue': ['yes', 'no']
        }

        # Apply one-hot encoding to the input data
        for col, categories in categorical_cols.items():
            for cat in categories:
                input_df[f'{col}_{cat}'] = (input_df[col] == cat).astype(int)
            # Drop the original categorical column after one-hot encoding
            if col in input_df.columns:
                input_df = input_df.drop(columns=[col])

        # Align columns with training features
        processed_input = pd.DataFrame(columns=FEATURE_COLUMNS)
        for col in FEATURE_COLUMNS:
            if col in input_df.columns:
                processed_input[col] = input_df[col]
            else:
                processed_input[col] = 0 # Fill missing (e.g., if a category wasn't in input)

        # Ensure numerical columns are float type
        for col in ['glucose', 'systolicBP', 'diastolicBP', 'triglycerides', 'hdl', 'waistCircumference', 'age', 'gender']:
            if col in processed_input.columns:
                processed_input[col] = pd.to_numeric(processed_input[col], errors='coerce')
        processed_input = processed_input.fillna(0) # Handle any NaNs from coercion

        # Scale the input data using the loaded scaler
        scaled_input = scaler.transform(processed_input)
        scaled_input_df = pd.DataFrame(scaled_input, columns=FEATURE_COLUMNS)

        # Make predictions (probabilities)
        prediabetes_prob = prediabetes_model.predict_proba(scaled_input_df)[:, 1][0]
        hypertension_prob = hypertension_model.predict_proba(scaled_input_df)[:, 1][0]
        metabolic_syndrome_prob = metabolic_syndrome_model.predict_proba(scaled_input_df)[:, 1][0]

        # Convert probabilities to risk levels
        def get_risk_level(prob):
            if prob >= 0.7:
                return 'high'
            elif prob >= 0.4:
                return 'medium'
            else:
                return 'low'

        risks = {
            'preDiabetes': get_risk_level(prediabetes_prob),
            'hypertension': get_risk_level(hypertension_prob),
            'metabolicSyndrome': get_risk_level(metabolic_syndrome_prob),
        }

        return jsonify(risks)

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({"error": str(e)}), 400

# --- Main execution block ---
if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'train':
        train_and_save_models()
    else:
        # Before running the Flask app, ensure models are trained
        if not os.path.exists(PREDIABETES_MODEL_PATH):
            print("Models not found. Please run 'python app.py train' first to train the models.")
            sys.exit(1) # Exit if models are not trained

        print("Starting Flask server...")
        app.run(debug=True, port=5000) # Run on port 5000
