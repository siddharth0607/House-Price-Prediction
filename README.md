# House Price Prediction

## Overview

This project involves predicting house prices for properties in Bengaluru, KA, based on various features using a machine learning model. The model is deployed as a web application using FastAPI and is containerized with Docker.

## Project Flow

1. **Model Training**: 
   - An XGBoost model is trained to predict house prices based on input features such as location, area type, and various numerical attributes.

2. **Deployment**: 
   - The trained model is saved as a pickle file and deployed with a FastAPI application. Docker is used to containerize the application.

3. **Web Application**: 
   - The FastAPI app provides a web interface where users can input feature values to get predictions.
   - The app includes two main endpoints:
     - **/**: Displays the home page with the input form.
     - **/predict**: Handles form submissions and displays the predicted house price.
     - **/predict_api**: Provides an API endpoint for getting predictions in JSON format.

## Running Locally

To run the application locally:

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/siddharth0607/House-Price-Prediction.git
   cd House-Price-Prediction

2. **Set Up a Virtual Environment**:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use venv\Scripts\activate

3. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt

4. **Build the Docker Image**:

   ```bash
   docker build -t house-price-app

5. **Run the Docker Container**:

   ```bash
   docker run --name house-price-container -p 5000:5000 house-price-app
   ```
   The app will be available at http://127.0.0.1:5000/

## Deployment

   To deploy your own version, ensure Docker is installed and follow the above local setup 
   steps. You can deploy using a cloud service that supports Docker containers.

## Contributing

   Contributions are welcome! Feel free to open issues or submit pull requests.
