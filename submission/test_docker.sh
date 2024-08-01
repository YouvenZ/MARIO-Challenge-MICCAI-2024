#!/bin/bash

docker build -t mario_inference .

# Run tests for Task 1
echo "Running Task 1 inference..."
sudo docker run --rm --gpus all -v $(pwd):/app  -v /path/to/dataset:/app/data -v $(pwd)/output:/app/output mario_inference python inference_pipeline_task_1.py
#sudo docker run --rm --gpus all -v /home/youven/Data/code/mario_challenge/submission:/app  -v /home/youven/Data/code/mario_challenge/data_challenge_docker/val_task1:/app/data -v $(pwd)/output:/app/output mario_inference python inference_pipeline_task_1.py

# Run tests for Task 2
echo "Running Task 2 inference..."
sudo docker run --rm --gpus all --v $(pwd):/app  -v /path/to/dataset:/app/data -v $(pwd)/output:/app/output mario_inference python inference_pipeline_task_2.py
#sudo docker run --rm --gpus all -v /home/youven/Data/code/mario_challenge/submission:/app  -v /home/youven/Data/code/mario_challenge/data_challenge_docker/val_task2:/app/data -v $(pwd)/output:/app/output mario_inference python inference_pipeline_task_2.py


echo "Tests completed. Check the output above for results."


-v /path/to/dataset:/app/data -v $(pwd)/output:/app/output