# Run all txt files in the benign and phishing folders through Trafilatura to extract the text from the html files
# Run extracted text through XLM-Roberta to get the embeddings
# Run extracted text first through Google Translate to translate to English and then through Sentence BERT to get the embeddings
# Run extracted and translated text through Electra to get the embeddings


# import trafilatura as tr
from transformers import XLMRobertaModel, XLMRobertaTokenizer, ElectraModel, ElectraTokenizer, AutoTokenizer, AutoModel
import torch
import numpy as np
import os
import re
import string
import time
import pickle
import json
import sys


# def extract_text():
#     """
#     Extract the text from the html files in the benign and phishing folders
#     :return: None
#     """
#     # Get the html files from the benign and phishing folders
#     benign_mislead_path = sys.argv[1]
#     benign_mislead_files = os.listdir(benign_mislead_path)
#     phishing_path = sys.argv[2]
#     phishing_files = os.listdir(phishing_path)
#     os.makedirs('extracted_benign_misleading_text', exist_ok=True)
#     os.makedirs('extracted_phishing_text', exist_ok=True)
#
#     print("*****benign_mislead_files*****", len(benign_mislead_files))
#
#     # Get the text from the benign and misleading html files
#     for file in benign_mislead_files:
#         try:
#             with open(benign_mislead_path + "/" + file, 'r', encoding="utf-8") as f:
#                 html = f.read()
#                 text = tr.extract(html)
#                 if text is None:
#                     continue
#                 with open('extracted_benign_misleading_text/' + file[:-4] + '.txt', 'w+', encoding="utf-8") as f:
#                     print("utf-8 ---", file[:-4])
#                     f.write(text)
#         except UnicodeDecodeError:
#             with open(benign_mislead_path + "/" + file, 'r', encoding="windows-1256") as f:
#                 html = f.read()
#                 text = tr.extract(html)
#                 if text is None:
#                     continue
#                 with open('extracted_benign_misleading_text/' + file[:-4] + '.txt', 'w+', encoding="utf-8") as f:
#                     print("windows-1256 ---", file[:-4])
#                     f.write(text)
#
#     print("*****phishing_files*****", len(phishing_files))
#     # Get the text from the phishing html files
#     for file in phishing_files:
#         try:
#             with open(phishing_path + "/" + file, 'r', encoding="utf-8") as f:
#                 html = f.read()
#                 text = tr.extract(html)
#                 if text is None:
#                     continue
#                 with open('extracted_phishing_text/' + file[:-4] + '.txt', 'w+', encoding="utf-8") as f:
#                     print("utf-8 ---", file[:-4])
#                     f.write(text)
#         except UnicodeDecodeError:
#             with open(phishing_path + "/" + file, 'r', encoding="windows-1256") as f:
#                 html = f.read()
#                 text = tr.extract(html)
#                 if text is None:
#                     continue
#                 with open('extracted_phishing_text/' + file[:-4] + '.txt', 'w+', encoding="windows-1256") as f:
#                     print("windows-1256 ---", file[:-4])
#                     f.write(text)

def create_embeddings():
    """
    Get the embeddings for the extracted text
    :return: None
    """
    create_xlm_roberta_embeddings()


def generate_xlm_roberta_embeddings_arrays(files, path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device, torch.cuda.get_device_name(0))
    model_name = 'xlm-roberta-base'
    tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)
    model = XLMRobertaModel.from_pretrained(model_name).to(device)
    model.eval().to(device)

    xlm_roberta_embeddings = np.array([]).reshape(0, 768)
    xlm_roberta_embeddings_labels = np.array([]).reshape(0, 1)

    print("-" * 100)
    for file in files:
        print(path + "/" + file)
        try:
            with open(path + "/" + file, 'r', encoding="utf-8") as f:
                text = f.read()
        except UnicodeDecodeError:
            with open(path + "/" + file, 'r', encoding="windows-1256") as f:
                text = f.read()
        encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = model(**encoded_input.to(device))
            sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask']).to(device)
            xlm_roberta_embeddings = np.concatenate((xlm_roberta_embeddings, sentence_embeddings.cpu().numpy()))
            if path == sys.argv[1]:
                xlm_roberta_embeddings_labels = np.concatenate(
                    (xlm_roberta_embeddings_labels, np.zeros((len(sentence_embeddings), 1))))

            else:
                xlm_roberta_embeddings_labels = np.concatenate(
                    (xlm_roberta_embeddings_labels, np.ones((len(sentence_embeddings), 1))))


    result = np.concatenate((xlm_roberta_embeddings, xlm_roberta_embeddings_labels), axis=1)
    return result, xlm_roberta_embeddings_labels
def create_xlm_roberta_embeddings():
    """
    Get the XLM-Roberta embeddings for the extracted text
    :return: None
    """
    # Get the extracted text from the benign and phishing folders
    benign_mislead_path = sys.argv[3]
    benign_mislead_files = os.listdir(benign_mislead_path)
    phishing_path = sys.argv[4]
    phishing_files = os.listdir(phishing_path)
    os.makedirs('embeddings', exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    result_legitimate, label_leg = generate_xlm_roberta_embeddings_arrays(
        benign_mislead_files, benign_mislead_path)
    print("result_legitimate shape", result_legitimate.shape)
    # print("label_leg shape", label_leg.shape)
    # print("label_leg", label_leg)
    result_phishing, label_phis = generate_xlm_roberta_embeddings_arrays(
        phishing_files, phishing_path)
    print("result_phishing shape", result_phishing.shape)
    # print("label_phis shape", label_phis.shape)
    # print("label_phis", label_phis)
    result = np.concatenate((result_legitimate, result_phishing), axis=0)

    print("result shape", result.shape)
    print("result", result[:-1, :])
    with open('embeddings/' + 'legitimate_embeddings-xlm-roberta.pkl', 'wb') as file:
        pickle.dump(result, file)


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

if __name__ == '__main__':
    # Parameters:
    # 1. Path to the benign and misleading html files
    # 2. Path to the phishing html files
    # 3. Path to the extracted benign and misleading text files
    # 4. Path to the extracted phishing text files
    # extract_text()
    create_embeddings()
