# thin-ML-deployment
Template for quickly deploying a ML endpoint 

cd to thin-ML-deployment  
uvicorn app.api:router  
curl -X POST “http://localhost:8000/v1/thinMLdeployment/predict" -H “accept: application/json” -H “Content-Type: application/json” -d “{\”data\”:[[0]]}”  
