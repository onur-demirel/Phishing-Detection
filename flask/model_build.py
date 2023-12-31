# Using the embeddings, build seperate models using the following:
# 1. XGBoost
# 2. CATBoost
# 3. ANN 3 layer
import os
import sys
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import pickle
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression


class ModelBuilder:

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

    def create_model(self, model_name, X_train, y_train):
        if model_name == "xgboost":
            model = XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=2, early_stopping_rounds=100)
            print("XGBoost model is training...")
            model.fit(X_train, y_train, verbose=True, eval_set=[(X_train, y_train)])
            pickle.dump(model, open("model/xgboost_model.pkl", "wb"))
        elif model_name == "catboost":
            # GPU training - uncomment the following lines to train on GPU (and comment the line above)
            # model = CatBoostClassifier(task_type="GPU", devices='0:1', iterations=2, learning_rate=1, depth=2)
            model = CatBoostClassifier(iterations=200, learning_rate=1, depth=2)
            print("CatBoost model is training...")
            model.fit(X_train, y_train, verbose=True, eval_set=[(X_train, y_train)], early_stopping_rounds=100)
            pickle.dump(model, open("model/catboost_model.pkl", "wb"))
        elif model_name == "ann":
            model = MLPClassifier(hidden_layer_sizes=(100, 50, 10), max_iter=1000, activation='relu', solver='adam', random_state=1, verbose=True)
            print("ANN model is training...")
            model.fit(X_train, y_train)
            pickle.dump(model, open("model/ann_model.pkl", "wb"))
        # elif model_name == "ensemble":
        #     estimators = [
        #         ('xgboost', XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=2, verbose=True)),
        #         ('catboost', CatBoostClassifier(iterations=200, learning_rate=1, depth=2, verbose=True)),
        #         ('ann', MLPClassifier(hidden_layer_sizes=(100, 50, 10), max_iter=1000, activation='relu', solver='adam',
        #                               random_state=1, verbose=True))
        #     ]
        #     clf = StackingClassifier(
        #         estimators=estimators, final_estimator=LogisticRegression()
        #     )
        #     print("Ensemble model is training...")
        #     clf.fit(X_train, y_train)
        #     pickle.dump(clf, open("model/ensemble_model.pkl", "wb"))
        else:
            print("Model name is not valid")
            exit(1)


    def load_model(self, model_name):
        if model_name == "xgboost":
            model = pickle.load(open("model/xgboost_model.pkl", "rb"))
        elif model_name == "catboost":
            model = pickle.load(open("model/catboost_model.pkl", "rb"))
        elif model_name == "ann":
            model = pickle.load(open("model/ann_model.pkl", "rb"))
        elif model_name == "ensemble":
            model = pickle.load(open("model/ensemble_model.pkl", "rb"))
        else:
            print("Model name is not valid")
            exit(1)
        return model



    def test_model(self, model, X_test, y_test):

        y_pred = model.predict(X_test)

        print(y_pred)
        print(y_test)
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("Precision:", precision_score(y_test, y_pred))
        print("Recall:", recall_score(y_test, y_pred))
        print("F1 Score:", f1_score(y_test, y_pred))


if __name__ == '__main__':
    os.makedirs('model', exist_ok=True)
    model_name = sys.argv[1] # xgboost, catboost, ann
    model_builder = ModelBuilder()
    X_train, X_test, y_train, y_test = model_builder.load_data()
    model_builder.create_model(model_name, X_train, y_train)
    loaded_model = model_builder.load_model(model_name)
    model_builder.test_model(loaded_model, X_test, y_test)




