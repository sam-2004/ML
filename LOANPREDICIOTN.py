import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import io

# Load dataset
ds = pd.read_csv('loan approval4.csv')
ds.columns = ds.columns.str.strip()

# Preprocess the data
label_encoders = {}
for column in ds.columns:
    if ds[column].dtype == 'object':
        le = LabelEncoder()
        ds[column] = le.fit_transform(ds[column])
        label_encoders[column] = le

# Splitting the dataset into features (X) and the target variable (y)
X = ds.iloc[:, 1:12]  # Features
y = ds['loan_status']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Streamlit app
st.title("Loan Approval Prediction")

# Create input fields for user input
input_data = {}
for column in X.columns:
    if ds[column].dtype == 'int64':
        input_data[column] = st.number_input(f"Enter {column}:", value=0, step=1)
    else:
        input_data[column] = st.text_input(f"Enter {column}:")

# Convert input data to DataFrame
input_df = pd.DataFrame([input_data])

# Ensure all object columns are transformed
for column in input_df.columns:
    if input_df[column].dtype == 'object':
        le = label_encoders.get(column)
        if le:
            try:
                input_df[column] = input_df[column].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
            except ValueError:
                st.error(f"Invalid input for {column}. Please enter a valid value.")
                st.stop()

# Make predictions
if st.button('Predict'):
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    if prediction[0] == 0:
        st.write("The loan application is likely to be rejected.")
    else:
        st.write("The loan application is likely to be approved.")
    st.write(f"Prediction probability: {prediction_proba[0][prediction[0]]:.2f}")

# Optionally display model accuracy and classification report
# Remove this block for deployment to avoid showing internal details
st.write(f"Model Accuracy: {accuracy:.2f}")
st.write("Classification Report:")
st.text(report)
