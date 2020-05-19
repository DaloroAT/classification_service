# Classification service
Service for image classification with PyTorch, Flask, Redis and Gunicorn. 

Classifier uses the Resnet18, trained on ImageNet (1000 classes). 

## Run
1) Install [Docker](https://docs.docker.com/engine/install/ubuntu/#installation-methods).
2) Install [Docker Compose](https://docs.docker.com/compose/install/).
3) Build and run services `docker-compose up --build`.

## How to use
The service provides an asynchronous API.  

1) Send POST request for classification \
`curl "http://0.0.0.0:8000/classify" -F "file=@my_image.jpg"` \
If the file is successfully read by the system, you will receive a response in which there is a `uuid` key.

2) Send GET request for checking result \
`curl "http://0.0.0.0:8000/classify?uuid=<your_uuid>"`\
Use `uuid` obtained on previous step, e.g. `curl "http://0.0.0.0:8000/classify?uuid=b48b2506-962c-4592-a480-ed7756b88dde"`. You will receive the result of the prediction, or the status of the request processing.
