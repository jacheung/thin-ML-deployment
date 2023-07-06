# POC ML Deployment
Thin deployment using FastAPI and Docker for a tensorflow transfer learning application on MNIST dataset. Can be easily configured for any type of machine learning model deployment.

This repository is part 1 of a series of templates for stepping up a POC deployment to one that is scalable in production. This part introduces:
1. FastAPI for building an inference point 
2. Abstracting out functions for best practices

## Background
We'll be performing transfer learning on the MNIST dataset to demo the tools used in this POC level workflow. The architecture diagram below will provide a high level view of what we'll be building.
![](/docs/architecture-poc.png)

## Usage

There are two ways to deploy an endpoint:
1. Local build using the CLI 
2. Docker deployment

#### 1. Local build
deploy FastAPI endpoint via:  
```
cd thin-ML-deployment  
uvicorn app.api:route   
```

test FastAPI endpoint via curl:  
```
curl \  
-F "file=@<test_image_file_path>" \  
http://127.0.0.1:8000/predict  
```

test FastAPI endpoint via python:  
```
#Python
url = "http://127.0.0.1:8000/predict"
filename = f'<test_image_file_path>'
file = {'file': open(filename, 'rb'}
resp = requests.post(url=url, files=file)
```

#### 2. Docker deployment 
1. Build the docker image 
``` 
cd thin-ML-deployment
docker build --file Dockerfile --tag thin-ml-deployment . 
```
2. Run the docker image
```
docker run -p 8000:8000 --name thin-ml-deployment
```
2a. (Optional) Enter the docker image
```
docker run -it --entrypoint /bin/bash thin-ml-deployment
```

3. Test the endpoint via curl:
```
curl \  
-F "file=@<test_image_file_path>" \  
http://0.0.0.0:8000/predict
```
