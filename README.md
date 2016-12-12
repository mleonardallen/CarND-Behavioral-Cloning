# Self-Driving Car Engineer Nanodegree: Behavioral Cloning

## Overview

In this project, a trained car drives in a simulated environment by cloning the behavior as seen during training mode.  Leveraging TensorFlow and Keras, a deep learning network predicts the proper steering angle given training examples.

## Dependencies

Install Python Dependencies with Anaconda (conda install …)
* numpy
* flask-socketio
* eventlet
* pillow
* h5py

Install Python Dependencies with pip (pip install ...)
* keras

## Files
* `model.py` - The script used to create and train the model.
* `drive.py` - The script to drive the car.
* `model.json` - The model architecture.
* `model.h5` - The model weights.

## Udacity Simulator

Udacity created a simulator based on the Unity engine that uses real game physics to create a close approximation to real driving.

### Download

* [Linux](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5831f0f7_simulator-linux/simulator-linux.zip)
* [macOS](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5831f290_simulator-macos/simulator-macos.zip)
* [Windows 32-bit](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5831f4b6_simulator-windows-32/simulator-windows-32.zip)
* [Windows 64-bit](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5831f3a4_simulator-windows-64/simulator-windows-64.zip)

Once you’ve downloaded it, extract it and run it.

### Run Server

Autonomous mode requires requires a server to receive steering commands.  Without the server not running, the car will just sit there in the simulated environment.

`python drive.py model.json`

## Network

### Approach

The README thoroughly discusses the approach taken for deriving and designing a model architecture fit for solving the given problem.

### Architecture

The README provides sufficient details of the characteristics and qualities of the architecture, such as the type of model used, the number of layers, the size of each layer. Visualizations emphasizing particular qualities of the architecture are encouraged.

## Training

The README describes how the model was trained and what the characteristics of the dataset are. Information such as how the dataset was generated and examples of images from the dataset should be included.


