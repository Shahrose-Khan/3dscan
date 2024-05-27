#!/bin/bash

# Check if port 8000 is already in use
if lsof -i:8000 -t >/dev/null; then
    echo "Port 8000 is already in use. Exiting."
    exit 1
else
    echo "Port 8000 is available. Starting the service..."

    # Activate the virtual environment
    source ../../private/3dscan_venv/bin/activate
    # If you're using conda, you might want to use the following line instead:
    # conda activate mesh3D

    # Run gunicorn with the specified parameters
    gunicorn -b 0.0.0.0:8000 -w 4 -t 1 c_code:app
fi
