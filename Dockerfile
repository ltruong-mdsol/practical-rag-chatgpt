FROM python:3.10

RUN apt-get update

COPY . /workspace

WORKDIR /workspace

RUN pip install -r requirements.txt

