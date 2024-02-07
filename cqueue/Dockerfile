FROM python:3.11.6

ENV CELERY_BROKER_URL redis://redis:6379/0
ENV CELERY_RESULT_BACKEND redis://redis:6379/0

COPY . /cqueue
WORKDIR /cqueue

RUN pip install -r requirements.txt

CMD ["celery", "-A", "tasks", "worker", "--loglevel=info"]