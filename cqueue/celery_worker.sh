#!/bin/bash

# Define a function to check if Celery is running
is_celery_running() {
  ps aux | grep "celery -A tasks worker" | grep -v grep > /dev/null 2>&1
  return $?
}

# Main script logic
if is_celery_running; then
  # Celery is running
  echo "Celery worker is currently running."
  read -p "Do you want to (s)top or (e)xit? (s/E) " choice
  case "$choice" in
    [Ss])
      echo "Stopping Celery worker..."
      pkill -f celery -A tasks
      ;;
    [Ee])
      echo "Exiting the script."
      exit 0
      ;;
    *)
      echo "Invalid choice. Please enter 's' or 'E'."
      ;;
  esac
else
  # Celery is not running
  echo "Celery worker is not currently running."
  read -p "Do you want to (s)tart or (e)xit? (s/E) " choice
  case "$choice" in
    [Ss])
      echo "Starting Celery worker..."
      celery -A tasks worker --loglevel=info
      ;;
    [Ee])
      echo "Exiting the script."
      exit 0
      ;;
    *)
      echo "Invalid choice. Please enter 's' or 'E'."
      ;;
  esac
fi

echo "Done!"
