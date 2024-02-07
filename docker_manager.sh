#!/bin/bash

# Function to display the menu and handle user input
function display_menu() {
    clear
    echo "Docker Management Script"
    echo "-----------------------"
    echo "1. Build Docker Images"
    echo "2. Start Docker Containers"
    echo "3. Stop and Remove Docker Containers"
    echo "4. Remove All Docker Images"
    echo "5. Quit"
    echo ""
    read -p "Enter your choice (1-5): " choice

    # Validate user input
    if [[ ! $choice =~ ^[1-5]$ ]]; then
        echo "Invalid choice. Please enter a number between 1 and 5."
        display_menu # Recursively call for another input
    else
        process_choice $choice
    fi
}

# Function to handle selected choice
function process_choice() {
    case $1 in
        1)
            echo "Building Docker Images..."
            docker-compose build --no-cache
            ;;
        2)
            echo "Starting Docker Containers..."
            docker-compose up -d
            ;;
        3)
            echo "Stopping and Removing Docker Containers..."
            docker-compose down -v
            ;;
        4)
            echo "Removing All Docker Images..."
            docker rmi $(docker images -q) # Remove all images, not just one named image_name
            ;;
        5)
            echo "Exiting script."
            exit 0
            ;;
    esac

    # Ask if the user wants to continue after each action
    read -p "Do you want to continue? (y/n): " continue_choice
    if [[ $continue_choice =~ ^[Yy]$ ]]; then
        display_menu
    fi
}

# Start the script by displaying the menu
display_menu

