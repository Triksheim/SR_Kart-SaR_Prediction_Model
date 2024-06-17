# SR Kart - SaR Prediction Model

This repository is an integral component of 'SR Kart,' a tool developed as part of a bachelor's thesis at UiT The Arctic University of Norway. The project is designed to enhance Search and Rescue (SAR) operations through technological solutions.

'SR Kart' comprises:
- A server backend that powers a web application. This backend manages data storage and performs data analysis and processing to create a prediction model.
- Mobile applications for both Android and iOS platforms, designed to interface seamlessly with the backend for real-time operational use.

## Map analysis and prediction model
The specific focus of this repository is on data processing and analysis. It involves collecting various geographical data for defined areas and analysing this data to develop a predictive model. This model aims to identify the most optimal search areas, enhancing the likelihood of locating missing persons based on environmental and probabilistic factors.
The model consists of three parts:
- Geographical data collection from various API's
- Data- preparation, processing and analysis
- Simulation for predicting high probability search areas


## Application Usage
This application is not intended to run as a standalone program but is designed to be triggered by function calls from the web application. The results are configured to display on the web and mobile platforms.


### Testing Locally
The model can be tested locally by running the script `web_test.py`, which simulates calls from a web application:
- Modify the parameters `lat`, `lng` to choose the center point of the search area.
- Adjust `d25`, `d50`, `d75` to define the overall search area size.
- The test script creates local directories for data management, logging, and generation of results.

### Geographical Limitations
National databases from GeoNorge are used for most of the geodata, therefore the program will not function outside of Norwegian land borders without modifications to utilize another data source.


# How It Works and Example Results

## Data Collection
This application collects various raster and vector data from Norwegian national databases via **GeoNorge** and crowd-sourced data from **OpenStreetMap** using the Overpass API. Data includes:
- Elevation data
- Ground types
- Vegetation
- Roads & paths
- Hiking trails
- Railways
- Buildings

## Data Processing
All collected data is compressed into a manageable size and rasterized. From the different raster data, a **terrain score matrix** is created. This matrix is normalized with cells containing values in the range of 0-1. The value represents the combined ‘score’ for each cell, indicating the difficulty of traversing through it. Easy terrain like flat trails are given high values close to 1, while rough terrain like steep slopes or lakes are assigned lower values.

Here is an example of a procsessed **Terrain Score Matrix**:

<img src="https://github.com/Triksheim/SR_Kart-SaR_Prediction_Model/assets/59808763/53327d6d-bd57-45e4-9b4d-40c38a7b90eb" width="600" alt="Terrain Matrix Image">\

## Simulation and Prediction
The simulation predicts areas with the highest probability of finding the lost person by simulating numerous potential routes a person could have taken. The simulation uses a branching algorithm, and the reach of each route from the starting point is calculated using the terrain score matrix. Each step a person takes on a route has a cost, determined by the terrain score, which is deducted from a movement resource pool. This approach results in challenging routes with rough terrain reaching shorter distances than routes on easier terrain.

Example of Simulation Results as search zones plotted on top of the **Terrain Score Matrix**:

<img src="https://github.com/Triksheim/SR_Kart-SaR_Prediction_Model/assets/59808763/6927ee8a-3dec-4bce-8869-669d20c92ef1" width="500" alt="Result Image">\

The increased probability search zones are created into GeoJSON polygon objects with corresponding coordinate reference system which can be overlayed on GIS applications or most standard map viewers.
Here are the map layers from the simulation shown as an overlay on a real map of the area in the GIS application QGIS:

<img src="https://github.com/Triksheim/SR_Kart-SaR_Prediction_Model/assets/59808763/97f7c9bf-2493-46df-a921-eafbe4ca3d87" width="800" alt="QGIS Image">\\

Logfile showing the different procsess that are executed during the models runtime and time taken for for each category:
<img src="https://github.com/Triksheim/SR_Kart-SaR_Prediction_Model/assets/59808763/bd2a3417-8177-4431-be90-bec2ffceb112" width="500" alt="logfile">\
