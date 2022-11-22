# Build Dash frontend image
docker build -t dash_frontend:v1 -f ./app/frontend/Dockerfile.frontend .
# Build Flask backend image
docker build -t flask_backend:v1 -f ./app/backend/Dockerfile.backend .
# Deploy both images and set up so containers so they can talk to one another using docker-compose
docker-compose up 
