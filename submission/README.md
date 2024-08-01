# Submission Instructions

## Overview

This repository contains the necessary files to help you dockerize your solution for the data challenge. Follow the steps below to build and run your Docker container for both tasks.

## Prerequisites

- Install [Docker](https://www.docker.com/get-started)

## Directory Structure

Ensure your submission follows this structure:

```
/submission
  ├── Dockerfile
  ├── requirements.txt
  ├── README.md
  ├── inference_task1.py
  ├── inference_task2.py
  ├── dataloader.py
  ├── test_docker.sh
  ├── validate_submission.sh
  ├── test_input/
  │   ├── before_slice.npy
  │   ├── after_slice.npy
  │   └── future_slice.npy
  ├── models/
  │   ├── example_model_task1.py
  │   ├── example_model_task2.py
  │   ├── model_task1.pth
  │   └── model_task2.pth
  └── utils/
      └── data_processing.py
```

## Building the Docker Image

1. Clone the repository or download the submission files.
2. Open a terminal and navigate to the directory containing the `Dockerfile`.
3. Build the Docker image:
   ```bash
   docker build -t oct_inference .
   ```

## Running the Docker Container

### Task 1

To run inference for Task 1:
```bash
docker run --rm -v $(pwd)/test_input:/app/input oct_inference python inference_task1.py
```

### Task 2

To run inference for Task 2:
```bash
docker run --rm -v $(pwd)/test_input:/app/input oct_inference python inference_task2.py
```

## Troubleshooting

- **Docker Build Errors**: Ensure your `requirements.txt` is correctly formatted and all dependencies are available.
- **File Not Found**: Ensure the file paths in your scripts match the mounted volumes.
- **Permission Issues**: Run Docker commands with appropriate permissions (use `sudo` if necessary).

## Notes

- Place your trained model files in the `models/` directory.
- Place your input data in the `input/` directory before running the container.
