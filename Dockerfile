FROM python:3.11.7

WORKDIR /app

ENV JAX_PLATFORM_NAME=cpu

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 3000
# Command to run your application (modify this based on your application's entry point)
CMD python tsnejax.py
