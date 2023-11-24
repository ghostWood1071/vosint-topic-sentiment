import gdown
import os
import sys

script_path = os.path.abspath(sys.argv[0])
running_path = os.path.dirname(script_path)

PATH_MODELS = os.path.join(running_path, "models")
NAME_MODEL ="bert_best_model.pt"
URL_MODEL = "https://drive.google.com/u/1/uc?id=1E-Gg33-WbJACK4pAYkcpPOuCGFZRUpnd&export=download"

if not os.path.exists(PATH_MODELS):
    os.mkdir(PATH_MODELS)

if not os.path.exists(os.path.join(PATH_MODELS,NAME_MODEL)):
    print('==== Download model ====')
    gdown.download(URL_MODEL,os.path.join(PATH_MODELS,NAME_MODEL),quiet=False)