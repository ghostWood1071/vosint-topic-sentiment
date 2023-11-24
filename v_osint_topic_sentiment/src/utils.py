import re
from nltk.tokenize import sent_tokenize
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