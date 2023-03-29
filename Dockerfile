FROM python:3.9.16-slim
ENV PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
COPY download_nltk_resources.py download_nltk_resources.py
COPY . /app
EXPOSE 8000
