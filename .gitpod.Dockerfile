# Use an existing base image with Python and necessary libraries
FROM python:3.8

# Install any additional system dependencies if needed
# For example, if you need to install OpenCV dependencies, you can do it here

# Set the working directory
WORKDIR /workspace

# Copy your project files into the workspace
COPY . .

# Install Python dependencies from requirements.txt
RUN pip install -r requirements.txt
