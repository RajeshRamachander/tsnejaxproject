#!/bin/bash

# Pull the Redis Docker image
docker pull redis

# Run a Redis container
docker run --name my-redis-container -p 6379:6379 -d redis

# Check if the container is running
if [ "$(docker inspect -f '{{.State.Running}}' my-redis-container 2>/dev/null)" == "true" ]; then
    echo "Redis container is running."
else
    echo "Failed to start Redis container."
fi
