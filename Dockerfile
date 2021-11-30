FROM python:3.8.12

MAINTAINER Moritz Alois Berner

WORKDIR /app

COPY './requirements.txt' .

# RUN apt-get install libgtk2.0-dev pkg-config -yqq 

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

COPY . .

CMD ["python", "app.py"]