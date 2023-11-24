FROM python:3.8-slim-buster

WORKDIR /home/app

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt &&\
  python -m nltk.downloader -d /usr/local/share/nltk_data punkt

COPY . .

EXPOSE 1510

CMD [ "python3","main.py"]

