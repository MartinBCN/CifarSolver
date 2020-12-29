FROM python:3.8

ENV DATA_DIR=/data
ENV MODEL_DIR=/models

COPY requirements.txt .

RUN pip install -r /requirements.txt

COPY . .

COPY /CifarClassifier .

EXPOSE 8000

CMD ["uvicorn", "main:app"]