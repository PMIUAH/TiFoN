# TiFoN (Timeseries Forecasting with normalization on Neural Networks)
TiFoN is a software designed to facilitate predictive time series analysis using window approach. It uses as input files with the time series data as sets of two-day records per row, with an hourly difference between each row. The software streamlines the configuration of neural networks, making setup more accessible and efficient.

This software is designed, implemented and delivered by members of de IMP (PMI) research group of the University of Alcala (Universidad de Alcalá), in Spain (España).

## Installation
To install the software, it is recommended to use Python version '3.12' or higher. After installing Python, it is recommended to follow the next steps:

1. Clone this repository: https://github.com/PMIUAH/TiFoN/tree/main to your computer.
2. Locate at the root of the project.
3. Update the pip package manager: `pip install --upgrade pip`
4. Run the following command: `pip install -r requirements.txt`
5. Once it is set at the `data_individual_sensors` or `data_collection_sensors` directories the train and test datasets from the sensors or elements from which you want to make the analysis, run the following command at the root of the project: `python main.py`.
