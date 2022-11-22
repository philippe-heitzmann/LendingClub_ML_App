docker build -t dash_frontend:v1 -f ./app/frontend/Dockerfile.frontend .
docker build -t flask_backend:v1 -f ./app/backend/Dockerfile.backend .
docker-compose up 
