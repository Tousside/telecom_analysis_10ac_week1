FROM node:18-alpine
# requirements
COPY requirements.txt

RUN pip install -r requirements.txt

WORKDIR . /app

CMD [ "python3", "/app/src/task5.py" ]
