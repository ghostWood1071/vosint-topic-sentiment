# import gdown
# import os


# PATH_MODELS = "./models"
# NAME_MODEL ="state_dict_sentiment_model.pt"
# NAME_W2V_TXT = "glove.6B.300d.txt"
# NAME_W2V_NPY = "glove.6B.300d.npy"
# NAME_LABEL_ENCODER = "classes.npy"
# # URL_MODEL = "https://drive.google.com/u/1/uc?id=1Syq9-WRBu2rvZeiZlYqR_lMD6E1HePBD&export=download"
# URL_MODEL = "https://drive.google.com/u/1/uc?id=19qkC19PYjxU5kb4oF2ku2UxZF_MhXDNp&export=download"
# URL_W2V_TXT = "https://drive.google.com/u/1/uc?id=1Tv-oDYfXse7X917jpNogqWC97sd71CL6&export=download"
# URL_W2V_NPY = "https://drive.google.com/u/3/uc?id=1-9RioHOf17yCIIJADgUVTKojJnfkPgRe&export=download"
# URL_LABEL_ENCODER = "https://drive.google.com/uc?export=download&id=1nPtraoPnGxIyPPXUxur66KbZzhKgExDJ"

# if not os.path.exists(PATH_MODELS):
#     os.mkdir(PATH_MODELS)

# if not os.path.exists(os.path.join(PATH_MODELS,NAME_LABEL_ENCODER)):
#     print('==== Download Label Encoder ====')
#     gdown.download(URL_LABEL_ENCODER,os.path.join(PATH_MODELS,NAME_LABEL_ENCODER),quiet=False)
    
# if not os.path.exists(os.path.join(PATH_MODELS,NAME_MODEL)):
#     print('==== Download model ====')
#     gdown.download(URL_MODEL,os.path.join(PATH_MODELS,NAME_MODEL),quiet=False)

# if not os.path.exists(os.path.join(PATH_MODELS,NAME_W2V_NPY)):
#     print('==== Download W2V ==== ')
#     gdown.download(URL_W2V_NPY,os.path.join(PATH_MODELS,NAME_W2V_NPY),quiet=False)
#     gdown.download(URL_W2V_TXT,os.path.join(PATH_MODELS,NAME_W2V_TXT),quiet=False)

