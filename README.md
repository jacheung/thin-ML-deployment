# thin-ML-deployment
Template for quickly deploying a ML endpoint 

deploy FastAPI endpoint via:  
```
cd to thin-ML-deployment  
uvicorn app.api:api   
```

test FastAPI endpoint via curl:  
```
curl \  
	-F "file=@<test_image_file_path>" \  
	http://127.0.0.1:8000/predict  
```

test FastAPI endpoing via python:  
```
#Python
url = "http://127.0.0.1:8000/predict"
filename = f'<test_image_file_path>'
file = {'file': open(filename, 'rb'}
resp = requests.post(url=url, files=file)
```
