# Run all txt files in the benign and phishing folders through Trafilatura to extract the text from the html files
# Run extracted text through XLM-Roberta to get the embeddings
# Run extracted text first through Google Translate to translate to English and then through Sentence BERT to get the embeddings
# Run extracted and translated text through Electra to get the embeddings


import trafilatura as tr
from transformers import XLMRobertaModel, XLMRobertaTokenizer, ElectraModel, ElectraTokenizer, AutoTokenizer, AutoModel
import torch
import numpy as np
import os
import pandas as pd
import re
import string
import time
import pickle
import json
import sys


def extract_text():
    """
    Extract the text from the html files in the benign and phishing folders
    :return: None
    """
    # Get the html files from the benign and phishing folders
    benign_mislead_path = sys.argv[1]
    benign_mislead_files = os.listdir(benign_mislead_path)
    phishing_path = sys.argv[2]
    phishing_files = os.listdir(phishing_path)
    os.makedirs('extracted_benign_misleading_text', exist_ok=True)
    os.makedirs('extracted_phishing_text', exist_ok=True)

    print("*****benign_mislead_files*****", len(benign_mislead_files))

    # Get the text from the benign and misleading html files
    for file in benign_mislead_files:
        try:
            with open(benign_mislead_path + "/" + file, 'r', encoding="utf-8") as f:
                html = f.read()
                text = tr.extract(html)
                if text is None:
                    continue
                with open('extracted_benign_misleading_text/' + file[:-4] + '.txt', 'w+', encoding="utf-8") as f:
                    print("utf-8 ---", file[:-4])
                    f.write(text)
        except UnicodeDecodeError:
            with open(benign_mislead_path + "/" + file, 'r', encoding="windows-1256") as f:
                html = f.read()
                text = tr.extract(html)
                if text is None:
                    continue
                with open('extracted_benign_misleading_text/' + file[:-4] + '.txt', 'w+', encoding="utf-8") as f:
                    print("windows-1256 ---", file[:-4])
                    f.write(text)

    print("*****phishing_files*****", len(phishing_files))
    # Get the text from the phishing html files
    for file in phishing_files:
        try:
            with open(phishing_path + "/" + file, 'r', encoding="utf-8") as f:
                html = f.read()
                text = tr.extract(html)
                if text is None:
                    continue
                with open('extracted_phishing_text/' + file[:-4] + '.txt', 'w+', encoding="utf-8") as f:
                    print("utf-8 ---", file[:-4])
                    f.write(text)
        except UnicodeDecodeError:
            with open(phishing_path + "/" + file, 'r', encoding="windows-1256") as f:
                html = f.read()
                text = tr.extract(html)
                if text is None:
                    continue
                with open('extracted_phishing_text/' + file[:-4] + '.txt', 'w+', encoding="windows-1256") as f:
                    print("windows-1256 ---", file[:-4])
                    f.write(text)

def create_embeddings():
    """
    Get the embeddings for the extracted text
    :return: None
    """
    create_xlm_roberta_embeddings()

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

    # Load the XLM-Roberta model and tokenizer
    model = XLMRobertaModel.from_pretrained('xlm-roberta-base')
    tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
    model.eval()
    print("*****model*****", model)

    xlm_roberta_embeddings = np.array([]).reshape(0, 768)
    xlm_roberta_embeddings_labels = np.array([]).reshape(0, 1)
    with open(benign_mislead_path + "/" + benign_mislead_files[2], 'r', encoding="utf-8") as f:
        text = f.read()
        print(benign_mislead_path + "/" + benign_mislead_files[2])
    encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
    print(len(encoded_input['input_ids'][0]))
    with torch.no_grad():
        model_output = model(**encoded_input)
        print(len(model_output[0]))
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        # xlm_roberta_embeddings = np.concatenate((xlm_roberta_embeddings, sentence_embeddings.cpu().numpy()))
        # xlm_roberta_embeddings_labels = np.concatenate((xlm_roberta_embeddings_labels, np.zeros((len(sentence_embeddings), 1))))


    # for file in benign_mislead_files:
    #     with open(benign_mislead_path + "/" + file, 'r', encoding="utf-8") as f:
    #         text = f.read()
    #     encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
    #     with torch.no_grad():
    #         model_output = model(**encoded_input)
    #         sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    #         # xlm_roberta_embeddings = np.concatenate((xlm_roberta_embeddings, sentence_embeddings.cpu().numpy()))
    #         # xlm_roberta_embeddings_labels = np.concatenate((xlm_roberta_embeddings_labels, np.zeros((len(sentence_embeddings), 1))))
    # print("*****xlm_roberta_embeddings*****", xlm_roberta_embeddings.shape)
    # print("*****xlm_roberta_embeddings_labels*****", xlm_roberta_embeddings_labels.shape)

    print("Sentence embeddings:")
    print(sentence_embeddings)
    print("Sentence embeddings shape:")
    print(sentence_embeddings.shape)

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

if __name__ == '__main__':
    # extract_text()
    create_embeddings()
