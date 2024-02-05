FROM python:3.11.6

WORKDIR /app

ENV JAX_PLATFORM_NAME=cpu

COPY . /app

RUN pip install --no-cache-dir --default-timeout=100 -r requirements.txt

EXPOSE 3000
# Command to run your application (modify this based on your application's entry point)
CMD python tsnejax.py
