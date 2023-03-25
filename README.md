# MapGenCMBS
Authors: Zach Wimpee and Zack Milam

# Overview
This project aims to build a model that takes in satellite image and topographic map data, and generates a corresponding output file which recreates the input location as a map in Combat Mission: Black Sea. The specific goal is to recreate Bakhmut in-game, a city located in Ukraine that has been the site of a major battle between Russian and Ukrainian forces since May 2022.

The model will use publicly available satellite and topographic map data from sources such as Google Earth and USGS to train the neural network. Additionally, street view data will be used to provide additional context to the model. By utilizing this data, we will be able to create a highly accurate map of the city of Bakhmut and its surrounding areas in the game.

# Project Structure
```markdown
data/
    satellite/
    topographic/
    streetview/
    map/
models/
src/
    data_collection/
    deployment/
    models/
    training/
    utils/
config.py
init.sh
LICENSE
README.md
requirements.txt
```
# Data
The data directory will contain the satellite, topographic, and street view data used to construct the training data, as well as the output map file. The satellite and topographic data will be downloaded from publicly available sources such as Google Earth and USGS. Street view data will be used to provide additional context to the model. The final output map file will be stored in the map subdirectory.

# Models
The models directory will contain the trained neural network model used to generate the map file.

# Source Code
The src directory will contain the source code for the project, organized into subdirectories:
- `data_collection`: scripts to collect and preprocess data from various sources, including satellite and topographic maps and street view images
- `models`: implementation of the deep learning model used to generate the map files
- `training`: scripts to train the deep learning model using the preprocessed data
- `deployment`: scripts to deploy the trained model and generate map files for specified locations
- `utils`: utility functions used throughout the project

# Data
The data directory will contain the preprocessed data necessary for the project. This will include the satellite and topographic map datasets used for training the model, as well as the street view images for each location used in the final model output.

# Models
The models directory will contain the trained deep learning model, saved in the appropriate format for deployment and generating map files.

# Maps
The maps directory will contain the generated map files for specified locations. For example, the file bakhmut.map would correspond to the map file generated for the city of Bakhmut.

# Historical Context
The battle for Bakhmut took place in the context of the ongoing conflict between Ukraine and Russian-backed separatists in eastern Ukraine. Beginning in May 2014, separatist forces took control of large parts of the Donetsk and Luhansk regions, including the city of Bakhmut, which was then known as Artemivsk. Ukrainian forces launched a major offensive in January 2015 to retake the city, which was an important transportation hub, but were repelled by separatist forces.

In May 2015, Ukrainian forces launched a second offensive, which succeeded in recapturing the city and surrounding areas. The fighting resulted in significant damage to the city's infrastructure and displacement of civilians. The conflict in eastern Ukraine has continued to the present day, with periodic flare-ups of violence and ongoing diplomatic efforts to resolve the conflict.

In 2022, the conflict between Ukraine and Russia escalated to a full-scale invasion, with Russian forces attempting to annex large portions of Ukrainian territory. One of the key battlegrounds has been the city of Bakhmut, located in eastern Ukraine. The city has been the site of a prolonged battle, with Russian forces attempting to capture it for nine months. The Russian military relied heavily on Wagner Group forces, including convicts and irregular formations from the Donetsk and Luhansk People’s Republic (DNR/LNR) militias, to make any advances in the battle for Bakhmut.

Despite these efforts, Russian forces have been unable to secure a significant victory in the city. Ukrainian forces have been successful in defending Bakhmut and surrounding areas, with significant fortifications undermining any tactical significance that capturing Bakhmut likely has for Russian forces. Furthermore, the Russian effort against Bakhmut does not further the Russian military’s operational or strategic battlefield aims.

The Ukrainian military has a window of opportunity to seize the battlefield initiative and launch a counteroffensive when the Russian effort around Bakhmut culminates either before or after taking the city. The likely imminent culmination of the Russian offensive around Bakhmut, the already culminated Russian offensive around Vuhledar, and the stalling Russian offensive in Luhansk Oblast are likely setting robust conditions for Ukrainian counteroffensive operations.

Combat Mission: Black Sea is a military simulation game that allows players to experience the tactics and challenges of modern warfare. By recreating the battle for Bakhmut in this game, we can analyze the tactical decisions made by both sides and examine how various factors such as terrain, equipment, and personnel have affected the outcome of the battle.

The current battle for Bakhmut in Ukraine is a critical front in the ongoing conflict between Russia and Ukraine. Russian forces have been attempting to capture the city since May 2022 but have faced significant resistance from Ukrainian forces. The battle has been characterized by heavy casualties and grinding human wave attacks by Russian forces against entrenched Ukrainian defenses.

By analyzing the battle for Bakhmut in Combat Mission: Black Sea, we can gain insights into how terrain and fortifications have affected the course of the battle, how different units and equipment have performed in combat, and how tactical decisions made by both sides have impacted the outcome of the battle.

To create a map of Bakhmut in Combat Mission: Black Sea, we will use satellite imagery, topographic maps, and street view datasets of the city to generate the terrain and layout of the city in the game. This map will then be populated with unit positions and equipment based on real-world reports and analysis of the battle.

Overall, analyzing the battle for Bakhmut in Combat Mission: Black Sea provides a unique opportunity to gain insights into the ongoing conflict between Russia and Ukraine and to better understand the tactical and strategic factors that have influenced the course of the battle.

# Data Sources
The satellite and topographic map datasets used in this project were sourced from publicly available resources. The satellite imagery was obtained from the European Space Agency's Sentinel Hub (https://sentinel-hub.com/), while the topographic maps were obtained from the National Oceanic and Atmospheric Administration's National Geophysical Data Center (https://www.ngdc.noaa.gov/). Street view images were sourced from Google Maps (https://www.google.com/maps). These datasets provided critical information about the terrain and features of the Bakhmut area, which helped to inform the creation of the map file for use in Combat Mission: Black Sea.

# Generating Map Files
The deep learning model implemented in this project takes as input a combination of satellite and topographic map data for a specified location, and generates a corresponding output file in Combat Mission: Black Sea's map file format. The deployment directory contains scripts for deploying the trained model and generating map files for specified locations.

The process of generating map files begins with data collection. As mentioned earlier, satellite imagery and topographic maps for the desired location are obtained from publicly available sources, and street view images are sourced from Google Maps. The collected data is preprocessed and prepared for training the deep learning model.

The deep learning model is responsible for learning the relationship between the input data and the corresponding output file format of Combat Mission: Black Sea. Once the model has been trained, it can be deployed to generate map files for new locations. The deployment directory contains scripts to deploy the trained model and generate map files for specified locations.

During the training process, the deep learning model learns to map the input satellite and topographic map data to the corresponding output file format of Combat Mission: Black Sea. This is done by feeding the model with pairs of input and output data and adjusting the model's parameters to minimize the difference between the predicted output and the actual output. This is done using a loss function, which measures the difference between the predicted output and the actual output. The model iteratively adjusts its parameters to minimize the loss function using an optimization algorithm, such as stochastic gradient descent.

In this project, the specific architecture and hyperparameters of the deep learning model used for training are not specified, but they will be designed to optimize the model's ability to learn the relationship between the input data and the desired output format. The model will be trained using a dataset that includes pairs of satellite and topographic map data and their corresponding Combat Mission: Black Sea map files. The training process will involve splitting the dataset into training and validation sets to assess the performance of the model during training and prevent overfitting to the training data. Once the model has been trained and validated, it can be deployed to generate map files for new locations.

To generate a map file, the deployment script takes in the desired location's latitude and longitude coordinates as input, along with any additional configuration variables, such as map size, terrain type, and weather conditions. The script then uses the trained model to generate a corresponding output file, which can be imported into Combat Mission: Black Sea to recreate the specified location as a playable map.

This process can be repeated for different locations, and the resulting maps can be used to analyze the terrain and plan military operations. Overall, the deep learning model and associated deployment scripts provide an accessible tool for generating accurate and realistic maps in Combat Mission: Black Sea, which can be used for training and simulation purposes.