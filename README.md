# Phishing Detection using Machine Learning
## Introduction
We have trained in total six models to detect phishing websites. The models are:
Using XLM-RoBERTa embeddings:
1. XGBoost Classifier
2. CatBoost Classifier
3. A three layer artificial neural network
Using SBERT embeddings:
1. XGBoost Classifier
2. CatBoost Classifier
3. A three layer artificial neural network

## Dataset
The dataset used for training the models is created using [Phishing Websites Dataset](https://sites.google.com/view/phishintention/experiment-structure#h.1v2qp1pagd81). 
Using html.txt files provided in the dataset, we then extracted the text from the txt files using [Trafilatura](https://pypi.org/project/trafilatura/). After that, we used [Sentence Transformers](https://pypi.org/project/sentence-transformers/) to get the embeddings of the text. The embeddings are then used to train the models.

## Results
The results of the models are as follows:

| Transformer Name | Model Name | Accuracy           | Precision          | Recall             | F1 Score           |
| ------------------ | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ |
| XLM-RoBERTa        | XGBoost | 0.9957571471865845 | 0.9960897653859232 | 0.9967676080299421 | 0.9964285714285716 |
| XLM-RoBERTa        | CatBoost | 0.991817355288413  | 0.9924111762676785 | 0.9936107753410465 | 0.9930106135128139 |
| XLM-RoBERTa        | ANN     | 0.9432265885442974 | 0.9304847986852917 | 0.9760386140320635 | 0.9527174827528185 |
| SBERT              | XGBoost | 0.979692704416031  | 0.9829632465543645 | 0.9808978032473734 | 0.9819294387608758 |
| SBERT              | CatBoost | 0.9679810895025249 | 0.9652340019102197 | 0.9775585219578254 | 0.971357170319108  |
| SBERT              | ANN     | 0.9803373804663157 | 0.977473249483762  | 0.9880455407969639 | 0.9827309615929036 |