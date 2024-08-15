from flask import Flask, request, jsonify
from flask import Flask
from flask_cors import CORS
from sklearn.preprocessing import StandardScaler
import pandas as pd
import pickle
import joblib

model = joblib.load('model.joblib')

def preprocess(df):
    #drop columns that are not relevant to our analysis
    preprocessed_df = df.drop(columns =['verification_status','emp_title','sub_grade','title','address','issue_d','pub_rec','earliest_cr_line'])
    
    #Deleting rows with null values
    columns_to_check = ['revol_util', 'mort_acc','pub_rec_bankruptcies','emp_length']
    preprocessed_df.dropna(subset=columns_to_check, inplace=True)

    # Delete highly correlated column (installment with loan_amount)
    preprocessed_df.drop(columns=['installment'],inplace=True)

    # identify columns with numerical datatypes
    numerical_data=preprocessed_df.select_dtypes(include='number')
    num_cols=numerical_data.columns

    # Remove outliers
    for col in num_cols:
        mean=preprocessed_df[col].mean()
        std=preprocessed_df[col].std()
    
        upper_limit=mean+3*std
        lower_limit=mean-3*std
    
        preprocessed_df=preprocessed_df[(preprocessed_df[col]<upper_limit) & (preprocessed_df[col]>lower_limit)]

    # Change the value of 'initial_list_status' to 0 and 1
    preprocessed_df['initial_list_status'] = preprocessed_df['initial_list_status'].replace({'w': 0, 'f': 1})

    # Change 'loan_status' values to 1 (Fully Paid) and 0 (Charged Off)
    preprocessed_df['loan_status'] = preprocessed_df['loan_status'].replace({'Fully Paid': 1, 'Charged Off': 0})

    # Convert categorical data to numeric with `pd.get_dummies`
    categorical_columns =["term","grade","emp_length","home_ownership","purpose","application_type"]

    df_dummies = pd.get_dummies(preprocessed_df[categorical_columns])

    # Merge the encoded columns back to the original DataFrame
    preprocessed_df = pd.concat([preprocessed_df, df_dummies], axis=1)

    # Drop the original categorical
    preprocessed_df.drop(categorical_columns, axis=1, inplace=True)

    # remove target variable
    preprocessed_df.drop(columns=["loan_status"], inplace=True)

    X_test = preprocessed_df.values
    # Create a StandardScaler instances
    scaler = StandardScaler()

    # Fit the StandardScaler
    X_scaler = scaler.fit(X_test)

    # Scale the data
    X_test_scaled = X_scaler.transform(X_test)

    return X_test_scaled


# Initialize Flask app
app = Flask(__name__)
CORS(app)  # This enables CORS for all routes

@app.route('/predict', methods=['POST'])
def predict_file_data():
    try:
        # Check if a file is part of the request
        if 'file' not in request.files:
            return jsonify({'error': 'No file part in the request'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # Read CSV file into a DataFrame
        df = pd.read_csv(file, low_memory=False)        
  
        # Apply preprocessing
        processed_features = preprocess(df)

        # Make prediction
        predictions = model.predict(processed_features)
        
        # Map the predictions to human-readable results
        result = ['Fully Paid' if pred == 1 else 'Charged Off' for pred in predictions]
        
        # Return the result
        return jsonify({'predictions': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/predict_loan_status', methods=['POST'])
def predict():
    data = request.get_json()  # Get data from POST request
    prediction = model.predict([data['features']])  # Predict using the model

    result = 'Fully Paid' if prediction == 1 else 'Charged Off'
    return jsonify({'prediction': result}) # Return the prediction

if __name__ == '__main__':
    app.run(debug=True)