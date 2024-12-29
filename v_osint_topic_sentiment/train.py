from transformers import BertModel, BertTokenizer
import torch
import torch.nn as nn
from .src.dataset import Dataset
from .src.BertClassifier import BertClassifier
import json
from .src.utils import get_evaluation
import numpy as np
import argparse
import os
from datetime import datetime
import warnings
from fastapi import FastAPI, File, UploadFile
from pymongo import MongoClient
from .logger import TaskLogger
import traceback
import os

# warnings.filterwarnings("ignore")

# def get_args( learning_rate: float, batch_size: int, num_epoch: int,
#     model_name: str, path_file_train: str, path_file_test: str,**kargs):
def get_args( learning_rate: float, batch_size: int, num_epoch: int,
    model_name: str, **kargs):
    parser = argparse.ArgumentParser(
        """Implementation of the model described in the paper: Hierarchical Attention Networks for Document Classification""")
    parser.add_argument("--batch_size", type=int, default=batch_size)
    parser.add_argument("--bert_name", type=str, default="NlpHUST/vibert4news-base-cased")
    parser.add_argument("--num_epoches", type=int, default=num_epoch)
    parser.add_argument("--lr", type=float, default=learning_rate)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--es_min_delta", type=float, default=0.0,
                        help="Early stopping's parameter: minimum change loss to qualify as an improvement")
    parser.add_argument("--es_patience", type=int, default=10,
                        help="Early stopping's parameter: number of epochs with no improvement after which training will be stopped. Set to 0 to disable this technique.")
    # parser.add_argument("--train_path", type=str, default=path_file_train)
    # parser.add_argument("--test_path", type=str, default=path_file_test)
    parser.add_argument("--num_class", type=str, default=3)
    parser.add_argument("--test_interval", type=int, default=1, help="Number of epoches between testing phases")
    parser.add_argument("--saved_path", type=str, default="v_osint_topic_sentiment/models")
    parser.add_argument("--max_token_length", type=int, default=70)
    parser.add_argument("--max_sent_length", type=int, default=35)
    parser.add_argument("--output_model_name", type=str, default=model_name)
    args = parser.parse_args()
    return args

def get_configs():
    config = {}
    try:
        with open(".env", mode="r") as f:
            lines = f.readlines()
            print(lines)
            for line in lines:
                splits = line.split("=")
                config[splits[0]] = ("=".join(splits[1:])).strip("\n").strip('"')
    except FileNotFoundError as err:
        config["MONGO_DETAILS"] = os.getenv("MONGO_DETAILS", None)
        config["DATABASE_NAME"] = os.getenv("DATABASE_NAME", None)
    except Exception as e:
        traceback.print_exc()
        raise e
    return config

def get_dataset(mongo_uri, db_name):
    client = MongoClient(mongo_uri)
    col = client[db_name]["sentiment_dataset"]
    dataset_projection = {
            "_id": {"$toString": "$_id"},
            "id_new": {"$toString": "$id_new"},
            "url": 1,
            "title": 1,
            "description": 1,
            "text": {"$ifNull": ["$text", "$content"]},
            "label": 1
        }
    dataset = {}
    dataset["train_set"] = [x for x in col.find(
        {"set_type": "train"}, 
        projection = dataset_projection)]
    dataset["test_set"] = [x for x in col.find(
        {"set_type": "test"}, 
        projection = dataset_projection)]
    client.close()
    return dataset
  
def train_model(opt):
    config = get_configs()
    # train_path = opt.train_path
    # with open(train_path, encoding='utf-8') as f:
    #     train_dataset = json.load(f)

    # test_path = opt.test_path
    # with open(test_path, encoding='utf-8') as f:
    #     test_dataset = json.load(f)
    opt.output_model_name = opt.output_model_name + datetime.now().strftime("_%d_%m_%Y-%H:%M:%S")
    dataset = get_dataset(config["MONGO_DETAILS"], config["DATABASE_NAME"])
    test_dataset = dataset["test_set"]
    train_dataset = dataset["train_set"]

    # Tạo mô hình BERT
    bert_model = BertModel.from_pretrained(opt.bert_name)
    tokenizer = BertTokenizer.from_pretrained(opt.bert_name)

    # Thiết lập tham số
    num_classes = opt.num_class
    batch_size = opt.batch_size
    number_not_good_epoch = 0
    best_acc = 0
    
    label_mapping = {'tieu_cuc': 0,
                    'trung_tinh': 1,
                    'tich_cuc': 2
                    }
    
    # Tạo mô hình phân lớp document
    model = BertClassifier(bert_model, num_classes)
    # Định nghĩa hàm mất mát và tối ưu hóa
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

    if torch.cuda.is_available():
        model.cuda()
    model.train()
    train = Dataset(tokenizer, train_dataset, label_mapping, opt.max_token_length, opt.max_sent_length)
    train_dataloader = torch.utils.data.DataLoader(
        train, batch_size=batch_size, pin_memory=True, shuffle=True, drop_last= True)

    test = Dataset(tokenizer, test_dataset, label_mapping, opt.max_token_length, opt.max_sent_length)
    test_dataloader = torch.utils.data.DataLoader(
        test, batch_size=batch_size)
    logs = []
    num_iter_per_epoch = len(train_dataloader)
    task_logger = TaskLogger(
        config["MONGO_DETAILS"],
        config["DATABASE_NAME"],
        "sentiment"
    )
    task_logger.set_task("task_logs")
    all_iter = num_iter_per_epoch * opt.num_epoches
    current_iter = 0
    try:
        for epoch in range(opt.num_epoches):
            epoch_log = {}
            epoch_log['epoch'] = epoch+1
            for iter, (ids, masks, num_sent, labels) in enumerate(train_dataloader):
                if torch.cuda.is_available():
                    device = torch.device("cuda:0")
                    labels = labels.to(device)
                    
                optimizer.zero_grad()
                logits = model(ids,masks,num_sent)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                if (iter+1)%2==0:
                    training_metrics = get_evaluation(labels.cpu().numpy(), logits.cpu().detach().numpy(), list_metrics=["accuracy"])
                    print("Epoch: {}/{}, Iteration: {}/{}, Lr: {}, Loss: {}, Accuracy: {}".format(
                    epoch + 1,
                    opt.num_epoches,
                    iter + 1,
                    num_iter_per_epoch,
                    optimizer.param_groups[0]['lr'],
                    loss,training_metrics['accuracy']))
                    epoch_log['train_loss'] = loss.cpu().detach().numpy().tolist()
                    epoch_log['train_accuracy'] = training_metrics['accuracy']
                    progress = round(((current_iter + 1)/all_iter)*100, 2)
                    task_logger.log_task({
                        "progress": progress,
                        "done": progress >=100
                    })
                current_iter+=1
            if iter % opt.test_interval == 0:
                model.eval()
                loss_ls = []
                te_label_ls = []
                te_pred_ls = []
                for ids, masks, num_sent, te_label in test_dataloader:
                    num_sample = len(te_label)
                    if torch.cuda.is_available():
                        ids = ids.cuda()
                        masks = masks.cuda()
                        te_label = te_label.cuda()
                    with torch.no_grad():
                        te_predictions = model(ids,masks,num_sent)
                    te_loss = criterion(te_predictions, te_label)
                    loss_ls.append(te_loss * num_sample)
                    te_label_ls.extend(te_label.clone().cpu())
                    te_pred_ls.append(te_predictions.clone().cpu())

                te_loss = sum(loss_ls) / test_dataset.__len__()
                te_pred = torch.cat(te_pred_ls, 0)
                te_label = np.array(te_label_ls)
                test_metrics = get_evaluation(te_label, te_pred.numpy(), list_metrics=["accuracy","loss", "confusion_matrix"])

                print("Epoch: {}/{}, Lr: {}, Loss: {}, Accuracy: {}".format(
                epoch + 1,
                opt.num_epoches,
                optimizer.param_groups[0]['lr'],
                te_loss, test_metrics["accuracy"]))
                epoch_log['valid_loss'] = te_loss.cpu().detach().numpy().tolist()
                epoch_log['val_accuracy'] = test_metrics["accuracy"]
                model.train()

                if best_acc < test_metrics['accuracy']:
                    best_acc = test_metrics['accuracy']
                    print("Best model is {}".format(test_metrics["accuracy"]))
                    torch.save(model.state_dict(), f"{opt.saved_path+os.sep+opt.output_model_name}.pt")
                    number_not_good_epoch = 0
            logs.append(epoch_log)
            number_not_good_epoch += 1
            if number_not_good_epoch > opt.es_patience > 0:
                print("Stop training at epoch {}. The lowest loss achieved is {}".format(epoch,best_acc))
                break
            progress = round((epoch+1)/opt.num_epoches, 2)
            task_logger.log_task({
                "progress": progress,
                "done": progress >=100
            })
        task_logger.log_task({
            "progress": 100,
            "done": True,
            "task_result": logs,
            "error": False
        })
    except Exception as e:
        traceback.print_exc()
        task_logger.log_task({"task_result": str(e), "error": True})
    finally:
        task_logger.close()
    return logs    

# if __name__ == "__main__":
#     opt = get_args()
#     train_model(opt)