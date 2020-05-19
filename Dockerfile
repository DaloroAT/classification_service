FROM ubuntu:18.04 as base

RUN apt-get update && apt install -y poppler-utils python3 python3-pip git libsm6 libxext6 libfontconfig1 libxrender1 wget

COPY ./requirements/requirements.txt .
RUN pip3 install -r requirements.txt

COPY ./classification_service /classification_service
COPY setup.py .
RUN pip3 install -e .

ENV PYTHONPATH /classification_service/:$PYTHONPATH
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV FLASK_APP=/classification_service/app/app.py

WORKDIR /classification_service/app


FROM base as classifier

COPY ./requirements/requirements_pytorch.txt .
RUN pip3 install -r requirements_pytorch.txt
