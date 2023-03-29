FROM python:3.9.16-slim
ENV TF_FORCE_GPU_ALLOW_GROWTH=false
ENV CUDA_VISIBLE_DEVICES=""
ENV TF_CPP_MIN_LOG_LEVEL=3
ENV PYTHONUNBUFFERED=1
ENV OMP_NUM_THREADS=1
ENV TF_NUM_INTEROP_THREADS=1
ENV TF_NUM_INTRAOP_THREADS=1
WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN python -c "from transformers import AutoTokenizer, AutoModel; tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased'); model = AutoModel.from_pretrained('bert-base-multilingual-cased')"
COPY download_nltk_resources.py download_nltk_resources.py
COPY . /app
EXPOSE 8000
