# thin-ML-deployment
Thin deployment using FastAPI and Docker for a tensorflow transfer learning application on MNIST dataset. Can be easily configured for any type of machine learning model deployment.

![](/docs/architecture.png)

## Local build  
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

## Docker deployment 
1. Build the docker image 
``` 
cd thin-ML-deployment
docker build --file Dockerfile --tag thin-ml-deployment . 
```
2. Run the docker image
```
docker run -p 8000:8000 thin-ml-deployment
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
