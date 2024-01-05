# Use an image of Python without GPU support
FROM python:3.11.7

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application code into the container
COPY . .

# Command to run your application (modify this based on your application's entry point)
CMD ["python", "tsnejax.py"]
