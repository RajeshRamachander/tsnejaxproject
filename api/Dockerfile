FROM python:3.11.6

ENV CELERY_BROKER_URL redis://redis:6379/0
ENV CELERY_RESULT_BACKEND redis://redis:6379/0

ENV HOST 0.0.0.0
ENV PORT 7020
ENV DEBUG true

WORKDIR /api
COPY . /api


RUN pip install -U setuptools pip
RUN pip install -r requirements.txt

EXPOSE 7020

CMD [ "flask", "run",  "--host=0.0.0.0", "--port=7020"]