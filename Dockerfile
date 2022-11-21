FROM python:3.9

WORKDIR /app
COPY . /app

RUN apt-get update && apt-get install -y sudo vim
RUN sudo apt install python3.9-distutils 

RUN pip install --upgrade pip
COPY requirements2.txt .
RUN pip install -r requirements2.txt

EXPOSE 8050
EXPOSE 5000

CMD ["uvicorn", "app:app", "--host=0.0.0.0" , "--reload" , "--port", "8000"]
# CMD gunicorn --bind 0.0.0.0:8050 app:server
# CMD ["python", "app.py"]
