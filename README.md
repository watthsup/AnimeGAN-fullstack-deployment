# Anime Face Converter

Anime Face Converter is a full-stack application that converts people's faces to an anime style using AnimeGAN. The application consists of three components: 
1. Front-end: A Streamlit-based user interface for the client to interact with the application.
2. Back-end: A FastAPI-based API app that processes the user's request and sends it to the inference server.
3. Inference server: A Triton Docker container that serves the AnimeGAN deep learning model and enhances concurrent usage.

The purpose of this application is to provide entertainment and make it easier to convert faces to anime style. It is also intended to be used as a portfolio project.

## Features

- End-to-end deployment app, making it easy for people to use.
- Utilizes the AnimeGAN deep learning model to convert people's faces to anime style.
- Uses Triton Docker container to enhance latency and throughput.
- The source code is well-organized and documented, making it easy to adapt to other deep learning projects.

## Requirements

- Python
- OpenCV
- Torch
- ONNX
- Triton
- NumPy
- FastAPI
- Streamlit

## Installation

To install and run the application, please follow the steps below:

1. Clone the repository to your local machine.
2. Install the required Python packages by running `pip install -r requirements.txt`.
3. Start the application by running `docker-compose up` in the root directory of the project.

## Usage

After following the installation steps above, you can use the application to convert your face to anime style. Access the front-end by navigating to `http://localhost:8501` in your web browser. Upload an image of your face, and the application will convert it to anime style using the AnimeGAN deep learning model.


## Special Thanks
Thanks to https://github.com/TachibanaYoshino/AnimeGANv2 for the AnimeGAN V1 & V2 pretrained weight

## Preview
![ezgif com-video-to-gif (4)](https://user-images.githubusercontent.com/121663706/222923746-01673fe3-1994-4adc-bd31-1d85de411669.gif)
