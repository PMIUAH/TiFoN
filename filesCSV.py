import os
import pandas as pd


def get_list_sensors_ids():
    """
    None --> []
    OBJ: EN: Get the list of sensors ids. ES: Obtener la lista de ids de los sensores.
    :return: en: List of sensors ids. es: Lista de ids de los sensores.
    """
    # EN: Get the list of sensors ids.
    # ES: Obtener la lista de ids de los sensores.
    list_sensors_ids = []
    filename = "sensors_ids.csv"

    # EN: Check if the file exists.
    # ES: Comprobar si el archivo existe.
    if not os.path.exists(filename):
        print("The file", filename, "does not exist.")
    else:
        # EN: Read the file.
        # ES: Leer el archivo.
        list_sensors_ids = pd.read_csv(filename)

    return list_sensors_ids


def get_individual_sensor_data(sensor_id):
    """
    int --> dataframe, dataframe
    OBJ: EN: Get the train and test datasets of a sensor in CSV files.
    ES: Obtener los datasets de entrenamiento y prueba de un sensor en archivos CSV.
    :param sensor_id: EN: Sensor id. ES: Id del sensor.
    :return: EN: Dataframes with the train and test datasets of a sensor.
    ES: Dataframes con los datasets de entrenamiento y prueba de un sensor.
    """
    # EN: Get the train and test datasets of a sensor in CSV files.
    # ES: Obtener los datasets de entrenamiento y prueba de un sensor en archivos CSV.
    filename = ('data_individual_sensors/Windows_Sliding_Training_Individual_Sensor_' + str(sensor_id) + '.csv')

    # EN: Check if the file exists.
    # ES: Comprobar si el archivo existe.
    if not os.path.exists(filename):
        print("The file", filename, "does not exist.")
        train_dataset = None
    else:
        # EN: Read the file.
        # ES: Leer el archivo.
        train_dataset = pd.read_csv(filename)

    filename = ('data_individual_sensors/Windows_Sliding_Test_Individual_Sensor_' + str(sensor_id) + '.csv')

    # EN: Check if the file exists.
    # ES: Comprobar si el archivo existe.
    if not os.path.exists(filename):
        print("The file", filename, "does not exist.")
        test_dataset = None
    else:
        # EN: Read the file.
        # ES: Leer el archivo.
        test_dataset = pd.read_csv(filename)

    return train_dataset, test_dataset


def get_collection_sensors_data():
    """
    None --> dataframe, dataframe
    OBJ: EN: Get the train and test datasets of the collection of sensors in CSV files.
    ES: Obtener los datasets de entrenamiento y prueba de la colección de sensores en archivos CSV.
    :return: EN: Dataframes with the train and test datasets of the collection of sensors.
    ES: Dataframes con los datasets de entrenamiento y prueba de la colección de sensores.
    """
    # EN: Get the train and test datasets of the collection of sensors in CSV files.
    # ES: Obtener los datasets de entrenamiento y prueba de la colección de sensores en archivos CSV.
    filename = 'data_collection_sensors/Windows_Sliding_Training_Collection_Sensors.csv'

    # EN: Check if the file exists.
    # ES: Comprobar si el archivo existe.
    if not os.path.exists(filename):
        print("The file", filename, "does not exist.")
        train_dataset = None
    else:
        # EN: Read the file.
        # ES: Leer el archivo.
        train_dataset = pd.read_csv(filename)

    filename = 'data_collection_sensors/Windows_Sliding_Test_Collection_Sensors.csv'

    # EN: Check if the file exists.
    # ES: Comprobar si el archivo existe.
    if not os.path.exists(filename):
        print("The file", filename, "does not exist.")
        test_dataset = None
    else:
        # EN: Read the file.
        # ES: Leer el archivo.
        test_dataset = pd.read_csv(filename)

    return train_dataset, test_dataset


def save_prediction_model_individual_sensor(sensor_id, model_structure, analysis_timestamp):
    """
    int, dataframe, String --> None
    OBJ: EN: Save the prediction model from a Neural Network used for a sensor.
    ES: Guardar el modelo de predicción de una Red Neuronal usado para un sensor.
    :param sensor_id: EN: Sensor ID. ES: ID del sensor.
    :param model_structure: EN: Model structure. ES: Estructura del modelo.
    :param analysis_timestamp: EN: Timestamp in which the analysis was made. ES: Marca de tiempo en la que
    se realizó el análisis.
    :return: None
    """
    filename = ('prediction_results_individual_equip/Prediction_Model_Individual_Sensor_' + str(sensor_id) + '_'
                + str(analysis_timestamp) + '.csv')
    model_structure.to_csv(filename, index=False)


def save_prediction_results_individual_sensor(sensor_id, prediction_results, analysis_timestamp):
    """
    int, dataframe, String --> None
    OBJ: EN: Save the comparison results of the prediction of a sensor in a csv file.
    ES: Guardar los resultados de comparación de la predicción de un sensor en un archivo csv.
    :param sensor_id: EN: Sensor ID. ES: ID del sensor.
    :param prediction_results: EN: Dataframe with the comparison results of the prediction.
    ES: Dataframe con los resultados de comparación de la predicción.
    :param analysis_timestamp: EN: Timestamp in which the analysis was made.
    ES: Marca de tiempo en la que se realizó el análisis.
    :return: None
    """
    filename = ('prediction_results_individual_equip/Prediction_Results_Individual_Sensor_' + str(sensor_id)
                + '_' + str(analysis_timestamp) + '.csv')
    prediction_results.to_csv(filename, index=False)


def save_prediction_model_collection_equips(model_structure, analysis_timestamp):
    """
    dataframe, String --> None
    OBJ: EN: Save the prediction model from a Neural Network used for a collection of sensors.
    ES: Guardar el modelo de predicción de una Red Neuronal usado para una colección de sensors.
    :param model_structure: EN: Model structure. ES: Estructura del modelo.
    :param analysis_timestamp: EN: Timestamp in which the analysis was made. ES: Marca de tiempo en la que
    se realizó el análisis.
    :return: None
    """
    filename = ('prediction_results_collection_equips/Prediction_Model_Collection_Sensors_' + str(analysis_timestamp)
                + '.csv')
    model_structure.to_csv(filename, index=False)


def save_prediction_results_collection_equips(prediction_results, analysis_timestamp):
    """
    dataframe, String --> None
    OBJ: EN: Save the comparison results of the prediction of a collection of sensors in a csv file.
    ES: Guardar los resultados de comparación de la predicción de una colección de sensors en un archivo csv.
    :param prediction_results: EN: Dataframe with the comparison results of the prediction.
    ES: Dataframe con los resultados de comparación de la predicción.
    :param analysis_timestamp: EN: Timestamp in which the analysis was made.
    ES: Marca de tiempo en la que se realizó el análisis.
    :return: None
    """
    filename = ('prediction_results_collection_equips/Prediction_Results_Collection_Sensors_' + str(analysis_timestamp)
                + '.csv')
    prediction_results.to_csv(filename, index=False)
