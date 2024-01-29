#!/bin/bash

# Define the port number
PORT=7020

# Function to start the Flask app
start_app() {
    echo "Starting Flask app...$PORT"
    FLASK_APP=server.py FLASK_ENV=development FLASK_DEBUG=0 /Users/rajeshramachander/Documents/GitHub/tsnejaxproject/venv/bin/python -m flask run --host=0.0.0.0 --port=$PORT
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
