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

elif [ "$1" == "stop" ]; then

    # Stop Redis
    echo "Stopping Redis..."

    # If Redis is still running, force kill it
    pkill -9 redis-server

    # Wait for Redis to stop forcefully
    while pgrep -f 'redis-server' > /dev/null; do
      sleep 1
    done

    echo "Redis stopped."

else
    echo "Usage: $0 [start|stop]"
fi
