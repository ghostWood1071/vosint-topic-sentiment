from fastapi import FastAPI, APIRouter
from pydantic import BaseModel
from v_osint_topic_sentiment.sentiment_analysis import topic_sentiment_classification, change_model
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import logging
import json
from v_osint_topic_sentiment.train import get_args, train_model
from v_osint_topic_sentiment.test import test_model
from fastapi import FastAPI, File, UploadFile, Form
from typing import Optional
import os 
import traceback
from threading import Thread

app = FastAPI()

router = APIRouter(
    prefix="",
    tags=["sentiment"],
    # dependencies=[Depends(get_token_header)],
)

origins = [
    "http://localhost",
    "http://localhost:4200",
    "http://api.aiacademy.edu.vn",
    "192.168.1.58:8002",
    "192.168.1.58",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    # allow_origins=origins,
    allow_origins="*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router, prefix="/api/sentiment")


logging.basicConfig(filename='./v_osint_sentiment.log',
                    level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S"
                    )

class Item(BaseModel):
    title: str
    description: str
    content: str

# class TrainerArguments():
#     def __init__(self, learning_rate: float=1e-12
#                  , batch_size: int=4, num_epoch: int=10, output_model_name:str="best_model_11_06_2024", path_file_train: str="", path_file_test: str="", val_size: float=0.1 ):
#         self.learning_rate = learning_rate
#         self.batch_size = batch_size
#         self.num_epoch = num_epoch
#         self.output_model_name = output_model_name
#         self.path_file_train = path_file_train
#         self.path_file_test = path_file_test
#         self.val_size = val_size

def on_success(data=None, message="success"):
    if data is not None:
        return {
            "message": message,
            "result": data
        }
    return {
        "message": message,
    }


def on_fail(message="fail"):
    return {
        "message": message,
    }


@app.post("/topic")
async def topic_sentiment_analysis(obj_input: Item):
    try:
        title = obj_input.title
        description = obj_input.description
        content = obj_input.content
        result = topic_sentiment_classification(
            title=title, description=description, content=content)
        str_log = obj_input.__dict__
        str_log["label"] = result
        str_log = json.dumps(str_log)
        logging.info(str_log)
        return on_success([result['sentiment_label'], result['topic_label']])
    except Exception as err:
        return on_fail(err)

@app.post("/train")
async def train_topic_sentiment_model(
                                    learning_rate: float = Form(...),
                                    batch_size: int = Form(...),
                                    num_epoch: int = Form(...),
                                    output_model_name: str = Form(...),
                                    # file_train: UploadFile = File(...), 
                                    # file_validation: UploadFile = File(...),
                                    ):
    try:
        # file_train_location = f"./files_upload/{file_train.filename}"
        # file_validation_location = f"./files_upload/{file_validation.filename}"
        # with open(file_train_location, "wb+") as f:
        #     f.write(file_train.file.read())

        # with open(file_validation_location, "wb+") as f:
        #     f.write(file_validation.file.read())
        # pagrams = get_args(learning_rate, batch_size, num_epoch, output_model_name, file_train_location, file_validation_location)
        pagrams = get_args(learning_rate, batch_size, num_epoch, output_model_name)
        result = Thread(target = train_model, kwargs = {"opt": pagrams})
        result.start()
        return on_success("training model ...")
    except Exception as err:
        traceback.print_exc()
        return on_fail(err)

@app.get("/get_model_name")
async def get_model_name():
    try:
        from v_osint_topic_sentiment.sentiment_analysis import CURRENT_MODEL
        files = os.listdir("v_osint_topic_sentiment/models")
        result = {}
        result['current_model'] = CURRENT_MODEL
        result['list_model']  = files
        return on_success(result)
    except Exception as e:
        return on_fail(e)

@app.post("/change_model")
async def _change_model(model_name: str = Form(...)):
    try: 
        change_model(model_name)
        return on_success()
    except Exception as e:
        return on_fail(str(e))

@app.post("/test_model")
async def _test_model(model_name: str = Form(...), 
                    file_test: UploadFile = File(...)):
    
    file_test_location = f"./files_upload/{file_test.filename}"
    with open(file_test_location, "wb+") as f:
        f.write(file_test.file.read())

    with open(file_test_location,'r', encoding='utf-8') as f:
        data_test = json.load(f)
    result = test_model(model_name, data_test)
    return on_success(result)

@app.post("/predict")
async def _predict(model_name: str = Form(...),
                   title: str = Form(...),
                   text: str = Form(...)):
    result = test_model(model_name, [{"title": title, "description": "", "text": text}])
    return on_success(result)

@app.post('/delete_model')
async def delete_model(model_name: str = Form(...)):
    try:
        path_file = os.path.join("v_osint_topic_sentiment/models",model_name)
        if os.path.exists(path_file):
            os.remove(path_file)
        return on_success("Xóa file thành công")
    except Exception as e:
        return on_fail(e)
    
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=1510)


