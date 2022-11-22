### Run manually with Docker

```
# Build Dash frontend
docker build -t dash_frontend:v1 -f ./app/frontend/Dockerfile.frontend .
docker run -it --rm --shm-size 1G -p 8050:8050 --name dash_frontend dash_frontend:v1
```

- See README.md for instructions on how to run entire app end to end with docker-compose
