# Real-Time Hand Gesture and Action Detection

This project implements real-time hand gesture and action recognition using Mediapipe for hand and face detection, TensorFlow for training a Long Short-Term Memory (LSTM) model, and OpenCV for video processing. The model predicts actions based on hand movements captured via webcam.

## Features

- **Real-time gesture and action recognition** with a pre-trained LSTM model.
- **Hand and face detection** using Google's Mediapipe.
- **Customizable training** for new gestures and actions.
- **Visualized probability** of each action class during detection.

## Table of Contents

- [Project Structure](#project-structure)
- [Setup](#setup)
- [Running the Project](#running-the-project)
- [Training the Model](#training-the-model)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)

## Project Structure

plaintext . ├── data/ │ ├── action.h5 # Pre-trained LSTM model weights │ ├── dataset/ # Dataset directory │ ├── sequences/ # Processed sequences for training ├── src/ │ ├── main.py # Main entry point │ ├── mediapipe_detection.py # Detection logic using Mediapipe │ ├── utils.py # Utility functions │ ├── models/ │ ├── lstm_model.py # LSTM model architecture and training script │ ├── config/ │ ├── config.yaml # Project configuration ├── requirements.txt # Python dependencies └── README.md # Project documentation


## Setup

Prerequisites:
Python 3.7 or higher
Git (optional, for cloning)
Webcam (for real-time gesture detection)

Step 1: Clone the Repository
bash git clone <repository-url> cd <repository-directory>


Step 2: Create and Activate a Virtual Environment (Optional)
bash

Create virtual environment
python -m venv venv

Activate virtual environment
On Windows:
venv\Scripts\activate

On macOS/Linux:
source venv/bin/activate


Step 3: Install Dependencies
Install the required Python libraries:
bash pip install -r requirements.txt


Step 4: Add the Pre-trained Model
Ensure action.h5 (pre-trained model) is in the data/ directory. If not, train the model by following the Training the Model section below.

## Running the Project

To start real-time action detection:
bash python src/main.py

This will start capturing video via your webcam and recognize hand gestures in real time.

Press q to quit.

## Training the Model

If you want to train your own LSTM model for new actions or gestures:

Step 1: Prepare the Dataset
Record videos of hand gestures and store them in data/dataset/.
Use mediapipe_detection.py to extract keypoints from the video frames.

Step 2: Train the Model
Run the training script:
bash python src/models/lstm_model.py

This will save the trained model as action.h5 in the data/ directory.

## Dependencies

Install project dependencies from requirements.txt. The key dependencies are:

mediapipe for hand and face detection
opencv-python for video processing
tensorflow for the LSTM model
numpy for numerical computations

Example of requirements.txt:
plaintext mediapipe==0.8.9 tensorflow==2.6.0 opencv-python==4.5.3.56 numpy==1.19.5 pyyaml==5.4.1


## Contributing

Fork the repository.
Create a feature branch: git checkout -b feature/my-feature
Commit your changes: git commit -m 'Add my feature'
Push to the branch: git push origin feature/my-feature
Open a pull request.

## License

This project is licensed under the MIT License. Feel free to use and modify the code.
Instructions
Replace <repository-url> with the actual GitHub URL.
You can update any section as needed to match your project specifics.
This README covers the project's setup, usage, and contribution guidelines, making it ready for hosting on GitHub. It provides clear instructions for users to set up and run the project, as well as contribute to it.
