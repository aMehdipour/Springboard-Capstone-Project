# Turbofan Remaining Engine Life Predictor
## Project Motivation and Background
This project aims to create a deep learning model that will predict the remaining life of a turbofan engine based on varying operational setting and sensor reading data. The dataset, provided by NASA Ames, was created using engine runs simulated in a framework called C-MAPSS and can be found [here](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps). The dataset used contains 100 different engines, each with an initial "health" value, and simulates sensor readings as their health degrades over numerous flights. The training dataset also contains each engine's remaining useful life (RUL) as "cycles" (number of remaining flights) until the engine reaches some specified efficiency threshold where we can consider the engine as failing. Being able to accurately predict the remaining life of of each engine can help reduce maintenance costs by allowing for preventative maintenance, and can even help prevent catastrophic failure. Additionally, a healthier engine is more efficient in flight, therefore being able to spot signs of efficiency loss can help save operating costs in terms of fuel and will be better for the environment.

## Model Choice
Many different types of models were tried and tested during the development phase for this project. Initially, work was done using a decision tree and the problem was set up as a classification problem, where the RUL was transformed into a binary "Maintenance required/not required" label once a certain threshold of remaining cycles was reached in the training dataset. The goal, however, was to predict the amount of remaining cycles for each engine at any given time based on sensor inputs, therefore a switch was made to using deep learning models and changing from a classification problem to a regression problem. Two initial deep learning models were made, a fully connected network and a LSTM network. 

In this initial test, the LSTM showed more promise than the fully connected net, but the performance of each was still lacking compared to other models tackling the same problem that were available open-source. The decision was made to try to improve model performance by changing each model to take input from an autoencoder in order to reduce the dimensionality of the features. A separate autoencoder was created and trained using a decoder. Once the encoder was trained, the training was frozen on the encoder section and it was then attached as the input to each model. After some further tuning of the network sizes and activation types on each layer, it was found that the fully connected net gave the best combination of performance and training time, thus it was the model selected for deployment.

## Deployment
Gradio was chosen as the deployment architecture for this project as it met all of the deployment requirements and provided and intuitive and aesthetically pleasing interface for the end user. The most impactful sensor readings were chosen as the user inputs, and are put on sliders that the user can vary. The user is also able to select a specific engine ID. The user inputs are combined with other sensor data from the provided dataset in order to give a full feature list to the model, and then are scaled and normalized and the output RUL is displayed in the application. The app is hosted on the HuggingFace servers, and can be found [here](https://huggingface.co/spaces/ArashMehdipour/Turbofan_Remaining_Life_Predictor)

## Requirements
Package requirements are listed in the supplied requirements.txt, but are repeated here for readability:

click==8.0.3
gradio==2.7.5.2
numpy==1.22.3
pandas==1.3.3
scikit_learn==1.0.2
tensorflow==2.8.0

## Code
The development work for the project can be found at my [GitHub page](https://github.com/aMehdipour/Springboard-Capstone-Project), while the final model scripts can be found [here](https://github.com/aMehdipour/Springboard-Capstone-Project/tree/main/ProjectPythonScripts). The data folder contains all of the data required for running the scripts, as well as the transformed and parsed output data required for training the model. The trained model is saved in the model/trainedModel folder, and all of the scripts can be found in the src folder. The app.py file is the application code for the Gradio application hosted on HuggingFace.

In order to run the code, please first make sure you have all of the packages installed that are listed in the requirements.txt file. The workflow for this project is ParseData.py -> Train.py -> Predict.py. The ParseData.py file will look for the appropriate data files in the folder you specify (i.e. ../data/train.txt) and perform the required data transformation and parsing and save the modified data as a .csv (i.e. trainingdata.csv) to the data folder. The Train.py file takes the modified data file location as an input, constructs and trains the autoencoder, and then assembles and trains the fully connected network. The NN is then finally saved into the model/trainedModels folder. Finally, the Predict.py file takes the path to this saved model and to the modified testing data generated by ParseData.py and uses that to perform predictions.

All of these files use the click package to support command line inputs for the file locations to be use in each respective script. 

### <span style="color:red">**NOTE**: </span> The click package command line inputs must be in string format **WITHOUT** quotes in order to work correctly. For example, the data is nominally in the data folder, and if I wanted to run the ParseData.py file on the train.txt file contained in data, the input I would give when prompted by the command line would be ../data/train.txt **NOT** '../data/train.txt' or "../data/train.txt". Adding quotes or any other punctuation will cause the code to fail. This is because click automatically converts the input text to a Python string, and inputting '../data/train.txt' in the command line will result in the input directory being  ''../data/train.txt'' which is invalid.

## Training Results

Training losses:

![](https://github.com/aMehdipour/Springboard-Capstone-Project/blob/main/training_loss.PNG)


Prediction results:

![](https://github.com/aMehdipour/Springboard-Capstone-Project/blob/main/predictions.PNG)
