#!/bin/bash

if [ "$1" == "start" ]; then
    # Start Redis in detached mode
    echo "Starting Redis..."
    redis-server --daemonize yes

    # Wait for Redis to start
    while ! nc -z localhost 6379; do
      sleep 1
    done

    echo "Redis has started."

    # Start Celery Worker (Replace 'your_app' with your actual app name) in detached mode
    echo "Starting Celery Worker..."
    celery -A cqueue.tasks.celery worker --loglevel=info --detach &

    # End of script
    echo "All services started successfully."

elif [ "$1" == "stop" ]; then
    # Stop Celery Worker
    echo "Stopping Celery Worker..."
    pkill -f 'celery -A app.server.celery worker'

    # Wait for Celery Worker to stop
    while pgrep -f 'celery -A app.server.celery worker' > /dev/null; do
      sleep 1
    done

    echo "Celery Worker stopped."

    # Stop Redis
    echo "Stopping Redis..."

    # If Redis is still running, force kill it
    pkill -9 redis-server

    # Wait for Redis to stop forcefully
    while pgrep -f 'redis-server' > /dev/null; do
      sleep 1
    done

    echo "Redis stopped."

    echo "All services stopped successfully."

else
    echo "Usage: $0 [start|stop]"
fi
