# %% [markdown]
# # Can We Predict Churn Rate?: The Lifeline of Startups and Their Customers

# %%
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import types
import random


# %%
# Step 1: Title and Intro
st.title('Can We Predict Churn Rate?: The Lifeline of Startups and Their Customers')
st.write("This app uses a logistics regression for KKBox's Churn Prediction Challenge.")

# %%
def my_hash_func(*args, **kwargs):
    return hash(args + tuple(kwargs.items()))

@st.cache(hash_funcs={types.FunctionType: my_hash_func})
def load_data():
    data = pd.read_csv('data/train_cleaned_st_deploy.csv')
    return data

data = load_data()

# %%
data.head()

# %%
# Preparing the data
msno = data['msno']
X = data.drop(['msno', 'is_churn'], axis=1)
y = data['is_churn']

# Splitting the data into training and test sets
X_train, X_test, y_train, y_test, msno_train, msno_test = train_test_split(X, y, msno, test_size=0.2, random_state=42)

# Scaling the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Training the model
model = LogisticRegression(penalty='l2', C=0.1, random_state=42, max_iter=1000)
model.fit(X_train_scaled, y_train)

# Creating a button to calculate probabilities
if st.button('Calculate Probabilities'):
    # Predicting probabilities on the test set
    y_proba = model.predict_proba(X_test_scaled)
    
    # Storing the results in a dataframe
    result_df = pd.DataFrame({'msno': msno_test, 'is_churn_proba': y_proba[:, 1]})


