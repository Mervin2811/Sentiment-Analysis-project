# Import required libraries
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import streamlit as st
import pandas as pd

# Corrected file path (use one of the following formats)
file_path = r"C:\Users\mervi\OneDrive\Desktop\TASK-03\archive (27)\IMDB Dataset.csv"  # Raw string
# file_path = "C:/Users/mervi/OneDrive/Desktop/TASK-03/archive (27)/IMDB Dataset.csv"  # Forward slashes
# file_path = "C:\\Users\\mervi\\OneDrive\\Desktop\\TASK-03\\archive (27)\\IMDB Dataset.csv"  # Escaped backslashes

# Load the dataset
dataset = load_dataset("csv", data_files=file_path)

# If the dataset does not have a "test" split, manually split it
if "test" not in dataset:
    dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)  # 80% train, 20% test

# Convert dataset to pandas DataFrame
train_data = pd.DataFrame(dataset["train"])
test_data = pd.DataFrame(dataset["test"])

# Inspect the dataset
print("Columns in the dataset:", train_data.columns)
print("First few rows of the dataset:")
print(train_data.head())

# Convert string labels to integers
label_mapping = {"negative": 0, "positive": 1}  # Map string labels to integers
train_data["sentiment"] = train_data["sentiment"].map(label_mapping)  # Apply mapping
test_data["sentiment"] = test_data["sentiment"].map(label_mapping)    # Apply mapping

# Prepare the data
X_train = train_data["review"].tolist()  # Features (text data)
y_train = train_data["sentiment"].tolist()  # Labels
X_test = test_data["review"].tolist()  # Features (text data)
y_test = test_data["sentiment"].tolist()  # Labels

# Convert text data to numerical features using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)  # Limit to 5000 features for simplicity
X_train_tfidf = vectorizer.fit_transform(X_train)  # Fit and transform training data
X_test_tfidf = vectorizer.transform(X_test)  # Transform test data

# Train a Logistic Regression model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_tfidf)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print(classification_report(y_test, y_pred))

# Streamlit app
st.title("Sentiment Analysis App")
st.write("Enter a movie review to analyze its sentiment.")

# Input text box
user_input = st.text_area("Enter your review here:")

if st.button("Analyze"):
    if user_input:
        # Convert user input to TF-IDF features
        user_input_tfidf = vectorizer.transform([user_input])
        
        # Predict
        prediction = model.predict(user_input_tfidf)[0]
        
        # Display result
        if prediction == 1:
            st.success("Positive Sentiment :)")
        else:
            st.error("Negative Sentiment :(")
    else:
        st.warning("Please enter a review.")