import re
from nltk.tokenize import sent_tokenize
import torch
from sklearn import metrics
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import torch

def preprocessing_text(text):
    text = re.sub("\r", "\n", text)
    text = re.sub("\n{2,}", "\n", text)
    text = re.sub("…", ".", text)
    text = re.sub("\.{2,}", ".", text)
    text.strip()
    return text

def preprocess(tokenizer,text,MAX_WORD_LENGTH, MAX_SENT_LENGTH):
    text = preprocessing_text(text)
    paragraphs = text.split("\n")
    sentences_ids = []
    sentences_mask = []
    sentences = []
    for paragraph in paragraphs:
        # lặp từng câu
        for sentence in sent_tokenize(text=paragraph):
            sentences.append(sentence)

    sentences_token = tokenizer(
        sentences, 
        max_length=MAX_WORD_LENGTH, 
        truncation=True, return_tensors="pt", 
        padding='max_length')
    
    sentences_ids = sentences_token['input_ids']
    sentences_mask = sentences_token['attention_mask']
    num_sent = len(sentences_ids)
    if len(sentences_ids) >= MAX_SENT_LENGTH:
        sentences_ids = sentences_ids[:MAX_SENT_LENGTH]
        sentences_mask = sentences_mask[:MAX_SENT_LENGTH]
        num_sent = MAX_SENT_LENGTH
    else:
        
        sentences_ids_padding = torch.zeros((MAX_SENT_LENGTH - len(sentences_ids),MAX_WORD_LENGTH),dtype=torch.long)
        sentences_ids = torch.concat((sentences_ids,sentences_ids_padding),0)
        sentences_mask_padding = torch.zeros((MAX_SENT_LENGTH - len(sentences_mask),MAX_WORD_LENGTH),dtype=torch.long)
        sentences_mask = torch.concat((sentences_mask,sentences_mask_padding),0)
    
    sentences_ids = sentences_ids.view(1,MAX_SENT_LENGTH,MAX_WORD_LENGTH)
    sentences_mask = sentences_mask.view(1,MAX_SENT_LENGTH,MAX_WORD_LENGTH)
    assert sentences_ids.size() == sentences_mask.size()
    return sentences_ids, sentences_mask, [num_sent]

def get_evaluation(y_true, y_prob, list_metrics):
    # encoder = LabelEncoder()
    # encoder.classes_= np.load("./dataset/plcd/classes.npy")
    y_pred = np.argmax(y_prob, -1)
    output = {}
    if 'accuracy' in list_metrics:
        output['accuracy'] = metrics.accuracy_score(y_true, y_pred)
    if 'loss' in list_metrics:
        try:
            output['loss'] = metrics.log_loss(y_true, y_prob)
        except ValueError:
            output['loss'] = -1
    if 'confusion_matrix' in list_metrics:
        # label_true = encoder.inverse_transform(y_true)
        # label_pred = encoder.inverse_transform(y_pred)
        label_true = y_true
        label_pred = y_pred
        output['F1'] = classification_report(label_true,label_pred)
        output['confusion_matrix'] = str(confusion_matrix(y_true, y_pred))
    return output

def matrix_mul(input, weight, bias=False):
    feature_list = []
    for feature in input:
        if feature.ndim == 1:
            feature = feature.view(1,-1)
        feature = torch.mm(feature, weight)
        if isinstance(bias, torch.nn.parameter.Parameter):
            feature = feature + bias.expand(feature.size()[0], bias.size()[1])
        feature = torch.tanh(feature).unsqueeze(0)
        feature_list.append(feature)
    return torch.cat(feature_list, 0).squeeze()

def element_wise_mul(input1, input2):
    feature_list = []
    for feature_1, feature_2 in zip(input1, input2):
        feature_2 = feature_2.unsqueeze(1).expand_as(feature_1)
        feature = feature_1 * feature_2
        feature_list.append(feature.unsqueeze(0))
    output = torch.cat(feature_list, 0)

    return torch.sum(output, 0).unsqueeze(0)