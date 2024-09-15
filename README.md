# Stock Price Prediction
This project aims to predict the close price of Apple stock for the following day using a dataset with five years, from 07/29/2019 to 07/26/2024, worth of Apple stock information. This project was done has a final project for a class on ML. 
The purpose of this project was implement a relatively new and advanced ML model using popular Python ML libraries, and understand the theory behind how these models work. I choose the topic of LSTM-RNNs. This repository contains my python script and 
the report that I wrote detailing what I learned, which includes some of the theory on LSTM-RNNs, how I built and trained my model, and an interpretation of my results.

## Table of Contents
Introduction
Features
Installation
Usage
Examples
Contributing
License
Contact

## Introduction
The dataset was downloaded from nasdaq.com, and is stored in and read from a Github repo https://github.com/BenTennyson4/stock-market-datasets. I use popular ML libraries, including Pandas, Numpy, Sklearn, and Tensorflow to create the train the model. The report 
contains valueable backgroud information that helps to understand how why LSTMs were developed, how they work, and how I built my model, and how I interpreted the results. This project is useful for anyone who wants to gain a basic yet essential 
understanding of LSTM networks, which was what I used it for.

## Features
Stock Price Prediction: The LSTM model predicts the next day's close price using historical stock price data.
Data Preprocessing: Includes data normalization using MinMaxScaler, removing currency symbols from the dataset, and transforming the data into a suitable format for LSTM input.
Sliding Window Cross-Validation: A fixed window size is used to train and validate the model across different time splits.
Model Architecture: The model includes two LSTM layers with 50 units each, dropout layers for regularization, and dense layers for the final prediction.
Hyperparameters: Configurable hyperparameters such as LSTM units, batch size, epochs, optimizer, and loss function.
Performance Metrics: The model is evaluated using metrics such as Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), R², and Explained Variance Score (EVS).

## Installation
Prerequisites
Make sure you have Python 3.x and the following libraries installed:
pandas
numpy
scikit-learn
tensorflow
matplotlib
You can install the required dependencies using the following command:
pip install -r requirements.txt

## Usage
Load and Preprocess Data: The dataset is automatically loaded and preprocessed by the script.

Run the Python Script:
python stock_price_prediction.py

The script will:
Train the LSTM model on historical stock data.
Predict the stock price for the next day.
Display evaluation metrics and visualizations of model performance.

Hyperparameters: The script uses predefined hyperparameters, but they can be adjusted in the script for experimentation.
LSTM Units: [50, 50]
Batch Size: 10
Epochs: 10
Optimizer: 'adam'
Loss Function: 'mean_squared_error'

Output: After running the script, the predicted price for the next day will be printed, and plots of training and validation loss as well as RMSE will be generated.

## License
This project is licensed under the MIT License – see the LICENSE file for details.

## Contact
Maintainer: Ben Tennyson
Email: benxy.tennyson@gmail.com
