### Run manually with Docker
```
# Build Flask backend
docker build -t flask_backend:v1 -f ./app/backend/Dockerfile.backend .
docker run -it --rm --shm-size 1G -p 5000:5000 --name flask_backend flask_backend:v1
```

- See README.md for instructions on how to run entire app end to end with docker-compose