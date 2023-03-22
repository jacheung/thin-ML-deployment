FROM python:3.9-slim-buster

WORKDIR /app
COPY requirements.txt requirements.txt
COPY ./app app

RUN apt-get -y update \
    && apt-get -y install curl \
    && apt-get install python3-dev python3-pip -y \
    && pip3 install -r requirements.txt

ENV PYTHONPATH=/app

EXPOSE 8000

ENTRYPOINT ["uvicorn"]
CMD ["app.api:route", "--host", "0.0.0.0"]