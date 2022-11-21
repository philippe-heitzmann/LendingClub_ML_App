FROM python:3.9

WORKDIR /app
COPY . .

RUN apt-get update && apt-get install -y sudo
 
RUN sudo apt install python3.9-distutils
RUN pip install -r requirements2.txt

CMD ["python", "app.py"]

