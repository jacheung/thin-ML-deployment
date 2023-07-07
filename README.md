# POC ML Deployment
 A viable product is defined as any deployed application that automates some process or improves the workflow for a human-in-the-loop. As a data scientist, it's easy to scrap together a notebook and toss it over the fence to data/machine learning engineers to productize. This line of thinking creates a large start-up cost for any POC to take off and become a viable product. This series of repositories will create a baseline process for transition but feel free to edit it with your desired toolsets. 

This repository is part 1 of a series of templates for stepping up a POC deployment to one that is scalable in production. This part introduces:
1. Abstracting out functions from notebooks
2. Deploying an inference point via FastAPI (local + docker deployment) 

![](/docs/vision-poc.png)

## Background
We'll be performing transfer learning on the MNIST dataset to demo the tools used in this POC level workflow. The architecture diagram below will provide a high level view of what we'll be building.
![](/docs/architecture-poc.png)

## Usage
### Setup
Clone this repository 
```
gh repo clone jacheung/thin-ML-deployment
```

Create virtual environment and install requirements into it.
```
python -m venv <venv/path/>
source <venv/path>/bin/activate
cd thin-ML-deployment
pip install requirements.txt
```

To train a model (assuming one doesn't exist already in app/ml/img_classifier):
```
source <venv/path>/bin/activate
cd thin-ML-deployment
python3 app/ml/model.py 
```

### Deployment
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
