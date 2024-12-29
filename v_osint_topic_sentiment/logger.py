from pymongo import MongoClient
from pymongo.database import Database
from pymongo.collection import Collection
from typing import Dict, Any

class TaskLogger:
    def __init__(self, mongo_uri:str, db_name:str, task_name:str):
        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]
        self.col = None
        self.task_name = task_name
    
    def set_task(self, col_name:str):
        self.col = self.db[col_name]
        exists = self.col.count_documents({"task": self.task_name})
        if exists:
            self.col.update_one(
                {"task": self.task_name}, 
                {"$set":{
                    "progress": 0,
                    "done": False,
                    "error": False
                }}
            )
            return
        self.col.insert_one({
            "task": self.task_name,
            "progress": 0,
            "done": False,
            "error": False
        })
        
    def log_task(self, log_detail:Dict[str, Any]):
        self.col.update_one(
            {"task": self.task_name}, 
            {"$set": log_detail}
        )
    
    def close(self):
        self.client.close()