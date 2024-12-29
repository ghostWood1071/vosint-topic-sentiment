import torch
from transformers import BertModel, BertTokenizer
from .src.BertClassifier import BertClassifier
import numpy as np
import nltk
import warnings 
from .src.utils import preprocess
import os
import sys
from sklearn.metrics import accuracy_score
from tqdm import tqdm

nltk.download('punkt')

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning) 

if torch.cuda.is_available():
    device = torch.device("cuda")
    print ("cuda")
else:
    device = torch.device("cpu")
    print ("cpu")

script_path = os.path.abspath(sys.argv[0])
running_path = os.path.dirname(script_path)

MODEL_PATH = os.path.join(running_path, "v_osint_topic_sentiment/models/bert_best_model.pt")
CURRENT_MODEL = "bert_best_model.pt"
BERT_NAME ="NlpHUST/vibert4news-base-cased"
MAX_SENT_LENGTH = 50
MAX_WORD_LENGTH = 100

LABEL_MAPPING = {
    0: "tieu_cuc",
    1: "trung_tinh",
    2: "tich_cuc"
                }


def test_model(model_name, data_test):
    bert_tokenizer = BertTokenizer.from_pretrained(BERT_NAME)
    bert_model = BertModel.from_pretrained(BERT_NAME)
    sentiment_model = BertClassifier(bert_model,num_classes=3)
    sentiment_model.load_state_dict(torch.load(os.path.join("v_osint_topic_sentiment/models",model_name),map_location=device))
    sentiment_model.eval()
    true_labels = []
    predict_labels = []

    for record in tqdm(data_test):
        text = record.get("title","")+'\n'+record.get("description","")+'\n'+record.get("text","")
        with torch.no_grad():           
            sentences_ids ,sentences_mask, num_sent = preprocess(bert_tokenizer,text,MAX_WORD_LENGTH, MAX_SENT_LENGTH)
            logits = sentiment_model(sentences_ids,sentences_mask,num_sent)
            logits = logits.cpu().detach().numpy()[0]
        index_pred = np.argmax(logits, -1)
        label_pred = LABEL_MAPPING[index_pred]
        predict_labels.append(label_pred)
        try:
            true_labels.append(record['label'][1])
        except Exception as e:
            pass

    return {
        "accuracy": accuracy_score(true_labels, predict_labels) if len(true_labels) else 0, 
        "predicts": predict_labels,
        "true_lables": true_labels 
    }
    