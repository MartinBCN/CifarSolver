FROM python:3.8

ENV DATA_DIR=/data
ENV MODEL_DIR=/models

COPY requirements.txt .

RUN pip install -r /requirements.txt

COPY . .

COPY /CifarClassifier .

EXPOSE 8000

CMD ["uvicorn", "model_api:app", "--host", "0.0.0.0", "--port", "8000"]
