#!/bin/bash

IMAGE_NAME="tsne-image"
CONTAINER_NAME="tsne-container"

function build_image() {
    echo "Building Docker image..."
    docker build --no-cache -t $IMAGE_NAME .
    if [ $? -eq 0 ]; then
        echo "Build successful."
    else
        echo "Build failed."
        exit 1
    fi
}

function run_container() {
    echo "Running Docker container..."
    docker run -d -p 3000:3000 --name $CONTAINER_NAME $IMAGE_NAME
    if [ $? -eq 0 ]; then
        echo "Container started successfully."
    else
        echo "Failed to start the container."
        exit 1
    fi
}

function stop_container() {
    echo "Stopping Docker container..."
    docker stop $CONTAINER_NAME
    docker rm $CONTAINER_NAME
    if [ $? -eq 0 ]; then
        echo "Container stopped and removed successfully."
    else
        echo "Failed to stop or remove the container."
    fi
}

function remove_image() {
    echo "Removing Docker image..."
    docker rmi $IMAGE_NAME
    if [ $? -eq 0 ]; then
        echo "Image removed successfully."
    else
        echo "Failed to remove the image."
    fi
}

PS3='Please enter your choice: '
options=("Build Image" "Run Container" "Stop and Remove Container" "Remove Image" "Quit")
select opt in "${options[@]}"
do
    case $opt in
        "Build Image")
            build_image
            ;;
        "Run Container")
            run_container
            ;;
        "Stop and Remove Container")
            stop_container
            ;;
        "Remove Image")
            remove_image
            ;;
        "Quit")
            break
            ;;
        *) echo "Invalid option $REPLY";;
    esac
done
