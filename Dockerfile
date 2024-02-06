# Use an existing Python image with Celery installed
FROM python:3.11.6

# Set the working directory inside the container
WORKDIR /myapp

# Install any dependencies your project requires
COPY requirements.txt /myapp/
RUN pip install -r requirements.txt

# Copy your project code into the container
COPY ./ /myapp/

# Define the command to run when the container starts
CMD celery -A app.server.celery worker --loglevel=info
