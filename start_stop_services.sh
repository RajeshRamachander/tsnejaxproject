#!/bin/bash

if [ "$1" == "start" ]; then
    # Start RabbitMQ in detached mode
    echo "Starting RabbitMQ..."
    /usr/local/opt/rabbitmq/sbin/rabbitmq-server -detached

    # Wait for RabbitMQ to start
    while ! nc -z localhost 5672; do
      sleep 1
    done

    echo "RabbitMQ has started."

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
    celery -A server.celery worker --loglevel=info --detach &

    # Optionally, start your Flask or Django application here if needed

    # End of script
    echo "All services started successfully."

elif [ "$1" == "stop" ]; then
    # Stop Celery Worker
    echo "Stopping Celery Worker..."
    pkill -f 'celery -A server.celery worker'

    # Wait for Celery Worker to stop
    while pgrep -f 'celery -A server.celery worker' > /dev/null; do
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

    # Stop RabbitMQ
    echo "Stopping RabbitMQ..."
    /usr/local/opt/rabbitmq/sbin/rabbitmqctl stop

    # Wait for RabbitMQ to stop
    while pgrep -f 'rabbitmq-server' > /dev/null; do
      sleep 1
    done

    echo "RabbitMQ stopped."

    echo "All services stopped successfully."

else
    echo "Usage: $0 [start|stop]"
fi
