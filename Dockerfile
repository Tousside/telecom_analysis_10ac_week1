FROM node:18-alpine
# requirements
COPY requirements.txt

RUN pip install -r requirements.txt

WORKDIR . /app

RUN mkdir /app/
