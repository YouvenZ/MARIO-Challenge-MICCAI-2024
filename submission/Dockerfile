# Use an official Python runtime as a parent image
FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install Python packages, install the package that you have used during dev
RUN pip install --no-cache-dir -r requirements.txt

# Make ports available to the world outside this container
EXPOSE 8080

RUN mkdir -p /app/output


# PLEASE WRITE YOUR TEAM NAME :)
ENV Team_name=Binks  

# Default command to run inference_task1.py
CMD ["python", "inference_task1.py"]
