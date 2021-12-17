FROM python:3.8.12

ENV PORT 80
EXPOSE 80

WORKDIR /app
COPY . .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD ["python", "app.py"]