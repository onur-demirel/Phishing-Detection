# Using the embeddings, build seperate models using the following:
# 1. XGBoost
# 2. CATBoost
# 3. ANN 3 layer

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
from xgboost import XGBClassifier

# Load the embeddings
with open('embeddings/legitimate_embeddings-xlm-roberta.pkl', 'rb') as f:
    embeddings = pickle.load(f)

def prepare_data(embeddings):
    # shuffle the embeddings
    np.random.shuffle(embeddings)

    labels = embeddings[:, -1]
    embeddings = embeddings[:, :-1]

    X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def xgboost_model(X_train, X_test, y_train, y_test):
    model = XGBClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("XGBoost Accuracy: ", (y_pred == y_test).mean())

def create_model():
    X_train, X_test, y_train, y_test = prepare_data(embeddings)
    xgboost_model(X_train, X_test, y_train, y_test)


if __name__ == '__main__':
    create_model()



