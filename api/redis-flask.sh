#!/bin/bash

# Define the port number
PORT=7020

# Function to start the Flask app
start_app() {
    echo "Starting Redis..."
    redis-server --daemonize yes

    # Wait for Redis to start
    while ! nc -z localhost 6379; do
      sleep 1
    done

    echo "Redis has started."

    echo "Starting Flask app...$PORT"
    FLASK_APP=./api/server.py FLASK_ENV=development FLASK_DEBUG=0 python -m flask run --host=0.0.0.0 --port=$PORT
    echo "Flask app started on port $PORT."
}

# Function to stop the Flask app
stop_app() {
    echo "Stopping Flask app..."
    # Find the Flask app process running on the specified port
    flask_pid=$(lsof -ti :$PORT)
    if [ -n "$flask_pid" ]; then
        kill "$flask_pid"
        echo "Flask app stopped."
    else
        echo "Flask app is not running on port $PORT."
    fi

    # Stop Redis
    echo "Stopping Redis..."

    # If Redis is still running, force kill it
    pkill -9 redis-server

    # Wait for Redis to stop forcefully
    while pgrep -f 'redis-server' > /dev/null; do
      sleep 1
    done

    echo "Redis stopped."
}

# Check the command-line argument
if [ "$1" == "start" ]; then
    start_app
elif [ "$1" == "stop" ]; then
    stop_app
else
    echo "Usage: $0 [start|stop]"
    exit 1
fi
