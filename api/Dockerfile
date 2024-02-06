FROM python:3.11.6

WORKDIR /app

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

COPY .. .

EXPOSE 7020

CMD [ "python", "./server.py" ]
