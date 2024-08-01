# Submission Instructions

Instructions for finalists of the MARIO CHALLENGE - MICCAI 2024

## Overview

This repository contains the necessary files to help you dockerize your solution for the data challenge. Follow the steps below to build and run your Docker container for both tasks. Update the `inference_pipeline_task_1.py` and `inference_pipeline_task_2.py` scripts according to your proposed approach. Please only change the mentioned parts of the scripts. For each task, you are allowed two submissions. We have provided examples of inference, ranging from simple inference to more complex methods such as model ensembling or test-time augmentation.

## Prerequisites

- Install [Docker](https://www.docker.com/get-started)

## Directory Structure

Ensure your submission follows this structure. You can change the name of the scripts inside the `models` folder as well as the names of the weights. They are provided as examples. If you need additional Python scripts, add them in the root or inside appropriate directories:

```
/submission
  ├── Dockerfile
  ├── requirements.txt
  ├── README.md
  ├── inference_pipeline_task_1.py
  ├── inference_pipeline_task_2.py
  ├── dataloader.py
  ├── test_docker.sh
  ├── csv/
  │   ├── task1.csv
  │   ├── task2.csv
  ├── models/
  │   ├── example_model_task1.py
  │   ├── example_model_task2.py
  │   ├── example_model_task1v2.py
  │   ├── example_model_task2v2.py  
  │   ├── model_task1.pth
  │   └── model_task2.pth
  │   ├── model_task1v2.pth
  │   └── model_task2v2.pth  
  └── utils/
      └── data_processing.py
```

## Building the Docker Image

1. Clone the repository or download the submission files.
2. Open a terminal and navigate to the directory containing the `Dockerfile`.
   1. Update the `requirements.txt` with your dependencies.
   2. We recommend using an official base image from PyTorch or TensorFlow, depending on the framework you used. In our example, we used a PyTorch-based image.
   3. Add your team name inside the Dockerfile where indicated.
   4. Place your trained models (scripts and weights files) in the `models/` directory.
   5. Place your `.csv` files for each task in the `csv/` directory before running the container.
3. Build the Docker image using the following command:
   ```bash
   docker build -t mario_inference .
   ```

## Running the Docker Container

### Task 1

To run inference for Task 1:
```bash
sudo docker run --rm --gpus all -v $(pwd):/app -v /path/to/dataset_task1:/app/data -v $(pwd)/output:/app/output mario_inference python inference_pipeline_task_1.py
```

### Task 2

To run inference for Task 2:
```bash
sudo docker run --rm --gpus all -v $(pwd):/app -v /path/to/dataset_task2:/app/data -v $(pwd)/output:/app/output mario_inference python inference_pipeline_task_2.py
```

### Full Pipeline

To run the full pipeline:
```bash
bash test_docker.sh
```

Once your test is successful, zip your submission folder and send it via the Google Form. The zip file should be named `MARIO_challenge_[TEAM_name].zip`.

## Troubleshooting

- **Docker Build Errors**: Ensure your `requirements.txt` is correctly formatted and that all dependencies are available.
- **File Not Found**: Ensure the file paths in your scripts match the mounted volumes.
- **Permission Issues**: Run Docker commands with appropriate permissions (use `sudo` if necessary).
- If you have questions or need help do not hesitate to contact me at rachid.zeghlache@univ-brest.fr 