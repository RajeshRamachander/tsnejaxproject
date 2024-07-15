#!/bin/bash

# Function to list running Docker containers
list_running_containers() {
    echo "Listing running Docker containers..."
    docker ps
}

# Function to start Redis container
start_redis() {
    echo "Starting Redis container..."

    # Check if the container already exists
    if [ "$(docker ps -aq -f name=redis-container)" ]; then
        echo "Redis container already exists. Removing the existing container..."
        docker rm -f redis-container || true
    fi

    # Check if the port is already in use
    if lsof -i:6379 -t >/dev/null; then
        echo "Port 6379 is already in use. Please free the port and try again."
        return
    fi

    # Start a new Redis container
    docker run -d --name redis-container -p 6379:6379 redis:latest

    # Check if the container started successfully
    if [ "$(docker inspect -f '{{.State.Running}}' redis-container)" = "true" ]; then
        echo "Redis container started successfully."
    else
        echo "Failed to start Redis container."
    fi
}

# Function to forcefully stop and remove Redis container
stop_redis() {
    echo "Stopping and removing Redis container..."
    if [ "$(docker ps -aq -f name=redis-container)" ]; then
        docker rm -f redis-container
        echo "Redis container forcefully stopped and removed."
    else
        echo "Redis container is not running."
    fi
}

# Display menu and handle user input
display_menu() {
    clear
    echo "Docker Management Script"
    echo "------------------------"
    echo "1. List Running Docker Containers"
    echo "2. Start Redis Container"
    echo "3. Stop and Remove Redis Container"
    echo "4. Quit"
    echo ""
    read -p "Enter your choice (1-4): " choice

    case $choice in
        1)
            list_running_containers
            ;;
        2)
            start_redis
            ;;
        3)
            stop_redis
            ;;
        4)
            echo "Exiting script."
            exit 0
            ;;
        *)
            echo "Invalid choice. Please enter a number between 1 and 4."
            ;;
    esac

    # Ask if the user wants to continue after each action
    read -p "Do you want to continue? (y/n): " continue_choice
    if [[ $continue_choice =~ ^[Yy]$ ]]; then
        display_menu
    else
        echo "Goodbye!"
        exit 0
    fi
}

# Start the script by displaying the menu
display_menu
