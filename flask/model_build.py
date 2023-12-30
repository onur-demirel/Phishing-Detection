# Using the embeddings, build seperate models using the following:
# 1. XGBoost
# 2. CATBoost
# 3. ANN 3 layer
import os

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
from xgboost import XGBClassifier

class ModelBuilder:
    def __init__(self):
        self.X_train, self.X_test, self.y_train, self.y_test = self.load_data()

    def load_data(self):
        # Load the embeddings
        with open('embeddings/legitimate_embeddings-xlm-roberta.pkl', 'rb') as f:
            embeddings = pickle.load(f)

        # shuffle the embeddings
        np.random.shuffle(embeddings)

        labels = embeddings[:, -1]
        embeddings = embeddings[:, :-1]

        X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test

    def xgboost_model(self, X_train, y_train):
        model = XGBClassifier()
        model.fit(X_train, y_train)
        pickle.dump(model, open("model/xgboost_model.pkl", "wb"))

    def load_model(self):
        model = pickle.load(open("model/xgboost_model.pkl", "rb"))
        return model

    def test_model(self, model, X_test, y_test):
        y_pred = model.predict(X_test)
        print(y_pred)
        print(y_test)
        print("Accuracy:", np.mean(y_pred == y_test))

    def create_model(self, X_train, y_train):
        self.xgboost_model(X_train, y_train)


if __name__ == '__main__':
    os.makedirs('model', exist_ok=True)
    model_builder = ModelBuilder()
    X_train, X_test, y_train, y_test = model_builder.load_data()
    model_builder.create_model(X_train, y_train)
    model = model_builder.load_model()
    model_builder.test_model(model, X_test, y_test)



