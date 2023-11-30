# Multi-modal Classifier Project

## Overview
This project implements a multi-modal classifier that integrates image and text data for classification tasks. Using neural network models, it processes and learns from both visual and textual inputs.

## Getting Started

### Prerequisites
- Python 3.8 or higher
- Docker (optional for containerized deployment)

### Installation
- Install Dependencies: Navigate to the project directory and install the required Python libraries.
    ```bash
    pip install -r requirements.txt
    ```
- Run the Application: Execute the main script to start the application.
    ```bash
    python main.py
    ```

### Project Structure
- File Structure of this project is as follows
    ```commandline
    multimodal_classifier_project/
    ├── data/
    │ ├── image_data.zip # Image data
    │ ├── image2text_train.csv # Training data
    │ └── image2text_test.csv # Test data
    ├── data_loader.py # Script for data loading and creating datasets
    ├── tokenizer.py # Script for text tokenization and preprocessing
    ├── model.py # Script containing neural network models
    ├── train.py # Script with training and validation functions
    ├── main.py # Main script to run the project
    ├── Dockerfile # Dockerfile for building the project image
    ├── .dockerignore # Specifies files/folders to ignore in Docker builds
    ├── requirements.txt # Required Python packages
    └── README.md # Project description and instructions
    ```
  

### Docker Deployment
- Build the Docker Image: From the project root, build the Docker image.
    ```bash
    docker build -t multimodal_classifier .
    ```
- Run the Docker Container: Once the build is complete, run the application in a Docker container.
    ```bash
    docker run -p 4000:80 multimodal_classifier
    ```

## Built With
- Python 3.8
- PyTorch
- Pandas
- NumPy
- NLTK

## License
This project is licensed under the MIT License.

## Acknowledgments
- Any confusion about the code please contact dliu328@wisc.edu
