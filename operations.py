import networkStructure
import filesCSV
import menu
import filesCSV as fcsv
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.metrics import (
    explained_variance_score,
    mean_absolute_error,
    mean_squared_error,
    mean_squared_log_error,
    median_absolute_error,
    r2_score,
    mean_absolute_percentage_error,
    mean_tweedie_deviance,
    max_error,
    d2_tweedie_score
)
from tensorflow.python.ops.metrics_impl import root_mean_squared_error

def get_sensors_ids_csv():
    """
    --> []
    OBJ: EN: Get the list of sensors IDs. ES: Obtiene la lista de IDs de sensores.
    :return: EN: List of sensors IDs. ES: Lista de IDs de sensores.
    """
    sensors_ids = fcsv.get_list_sensors_ids().values.transpose().tolist()[0]

    # EN: Remove the last element of the list as that sensor only has a record.
    # ES: Elimina el último elemento de la lista ya que ese sensor solo tiene un registro.
    sensors_ids.pop()
    return sensors_ids


def get_data_individual_sensor_csv(sensor_id):
    """
    int --> list
    OBJ: EN: Get the data of an individual sensor. ES: Obtener los datos de un sensor individual.
    :param sensor_id: EN: Sensor ID. ES: ID del sensor.
    :return: list EN: List of the data of the sensor. ES: Lista de los datos del sensor.
    """
    # EN: Get the data of the sensor.
    # ES: Obtener los datos del sensor.
    data_sensor = fcsv.get_individual_sensor_data(sensor_id)

    return data_sensor


def get_data_collection_sensors_csv():
    """
    --> list
    OBJ: EN: Get the data of the collection of sensors. ES: Obtener los datos de la colección de sensores.
    :return: list EN: List of the data of the collection of sensors. ES: Lista de los datos de la colección de sensores.
    """
    # EN: Get the data of the collection of sensors.
    # ES: Obtener los datos de la colección de sensores.
    data_sensors = fcsv.get_collection_sensors_data()

    return data_sensors


def network_execution_individual_sensor(model, activation_functions, number_neurons_layers, epochs=100):
    """
    model, list, list, int --> DataFrame
    OBJ: EN: Execute the network every time with the data from an individual sensor every time. ES: Ejecutar la red
    cada vez con los datos de un sensor individual cada vez.
    :param model: EN: Model of the network. ES: Modelo de la red.
    :param activation_functions: EN: List of the activation functions for each layer. ES: Lista de las funciones de
    activación para cada capa.
    :param number_neurons_layers: EN: List of the number of neurons for each layer. ES: Lista del número de neuronas
    para cada capa.
    :param epochs: EN: Number of epochs for the training of the model. ES: Número de épocas para el entrenamiento del
    modelo.
    """
    #TODO
    # EN: Ask the user for the sensors IDs to be analyzed.
    # ES: Preguntar al usuario por los IDs de los sensores a analizar.
    sensors_ids = get_sensors_ids_csv()
    metrics_results = pd.DataFrame()

    #TODO
    # EN: Timestamp in which the analysis started. ES: Marca de tiempo en la que comenzó el análisis.

    for sensor_id in sensors_ids:
        #TODO
        # EN: Build as a dataframe the information of the model to save it in a CSV file.
        # ES: Construir como un dataframe la información del modelo para guardarlo en un archivo CSV.
        # model_dataframe = networkStructure.build_model_network_dataframe(
        #     activation_functions, number_neurons_layers, epochs)
        #TODO
        # EN: Save the information of the model in a CSV file.
        # ES: Guardar la información del modelo en un archivo CSV.
        # filesCSV.save_prediction_model_individual_equipment(sensor_id, model_dataframe)


        # EN: Get the data of the equipment from the CSV file.
        # ES: Obtener los datos del equipo del archivo CSV.
        list_data = get_data_individual_sensor_csv(sensor_id)

        # EN: Execute the network with the data of the equipment.
        # ES: Ejecutar la red con los datos del equipo.
        metrics = network_execution(list_data, model, epochs=epochs)
        print(metrics)

        # EN: Transpose the dataframe to have the metrics in columns.
        # ES: Transponer el dataframe para tener las métricas en columnas.
        metrics = metrics.transpose()

        # EN: Add at the first column the sensor ID.
        # ES: Agregar en la primera columna el ID del sensor.
        metrics.insert(0, "Sensor ID", sensor_id)

        # EN: Add the metrics to the dataframe with the results. If the dataframe is empty, set the metrics as the
        # first row of the dataframe.
        # ES: Agregar las métricas al dataframe con los resultados. Si el dataframe está vacío, establecer las
        # métricas como la primera fila del dataframe.
        if metrics_results.empty:
            metrics_results = metrics
        else:
            metrics_results = pd.concat([metrics_results, metrics])

        #TODO
        # EN: Save the metrics of the network in a CSV file.
        # ES: Guardar las métricas de la red en un archivo CSV.
        # filesCSV.save_prediction_results_individual_sensor(sensor_id, metrics_results)

        # EN: Reset the dataframe with the results to be empty for the next sensor.
        # ES: Reiniciar el dataframe con los resultados para que esté vacío para el siguiente sensor.
        metrics_results = pd.DataFrame()


def network_execution_collection_sensors(model, activation_functions, number_neurons_layers, epochs=100):
    """
    model, list, list, int --> DataFrame
    OBJ: EN: Execute the network with the data of the collection of sensors. ES: Ejecutar la red con los datos de la
    colección de sensores.
    :param model: EN: Model of the network. ES: Modelo de la red.
    :param activation_functions: EN: List of the activation functions for each layer. ES: Lista de las funciones de
    activación para cada capa.
    :param number_neurons_layers: EN: List of the number of neurons for each layer. ES: Lista del número de neuronas
    para cada capa.
    :param epochs: EN: Number of epochs for the training of the model. ES: Número de épocas para el entrenamiento del
    modelo.
    """
    metrics_results = pd.DataFrame()

    #TODO
    # EN: Timestamp in which the analysis started. ES: Marca de tiempo en la que comenzó el análisis.

    #TODO
    # EN: Build as a dataframe the information of the model to save it in a CSV file.
    # ES: Construir como un dataframe la información del modelo para guardarlo en un archivo CSV.
    # model_dataframe = networkStructure.build_model_network_dataframe(activation_functions, number_neurons_layers,
    #                                                                  epochs)

    #TODO
    # EN: Save the information of the model in a CSV file.
    # ES: Guardar la información del modelo en un archivo CSV.
    # filesCSV.save_prediction_model_collection_equips(model_dataframe, init_time)

    # EN: Get the data of the sensors from the CSV file.
    # ES: Obtener los datos de los equipos del archivo CSV.
    list_data = get_data_collection_sensors_csv()

    # EN: Execute the network with the data of the sensors.
    # ES: Ejecutar la red con los datos de los sensors.
    metrics = network_execution(list_data, model, epochs=epochs)
    print(metrics)

    # EN: Transpose the dataframe to have the metrics in columns.
    # ES: Transponer el dataframe para tener las métricas en columnas.
    metrics = metrics.transpose()

    # EN: Add the metrics to the dataframe with the results. If the dataframe is empty, set the metrics as the first
    # row of the dataframe.
    # ES: Agregar las métricas al dataframe con los resultados. Si el dataframe está vacío, establecer las métricas
    # como la primera fila del dataframe.
    if metrics_results.empty:
        metrics_results = metrics
    else:
        metrics_results = pd.concat([metrics_results, metrics])

    #TODO
    # EN: Save the metrics of the network in a CSV file.
    # ES: Guardar las métricas de la red en un archivo CSV.
    # filesCSV.save_prediction_results_collection_equips(metrics_results, init_time)


def network_execution(list_data, model, epochs=100):
    """
    list, model, int --> list
    OBJ: EN: Execute the network with the data of the sensor. ES: Ejecutar la red con los datos del sensor.
    :param list_data: EN: List of the data of the sensor. ES: Lista de los datos del sensor.
    :param model: EN: Model of the network. ES: Modelo de la red.
    :param epochs: EN: Number of epochs for the training of the model. ES: Número de épocas para el entrenamiento del
    modelo.
    :return: list EN: List of the predictions of the network. ES: Lista de las predicciones de la red.
    """
    # EN: Set init time for the network. ES: Establecer el tiempo de inicio para la red.
    init_time = pd.Timestamp.now()

    # EN: Get from the list the data of the equipment the first dataframe as training dataset and the second dataframe
    # as testing dataset.
    # ES: Obtener de la lista los datos del equipo el primer dataframe como conjunto de datos de entrenamiento y el
    # segundo dataframe como conjunto de datos de prueba.
    training_dataset = list_data[0]
    testing_dataset = list_data[1]

    # EN: Separate the all the columns except the last one to be used as features to get the output of the network, and
    # the last column to be used as the output of the network.
    # ES: Separar todas las columnas excepto la última para ser usadas como características para obtener la salida de la
    # red, y la última columna para ser usada como la salida de la red.
    training_X = training_dataset.iloc[:, :-1].values
    training_y = training_dataset.iloc[:, -1].values
    testing_X = testing_dataset.iloc[:, :-1].values
    testing_y = testing_dataset.iloc[:, -1].values

    # EN: Execute the network with the data of the equipment. ES: Ejecutar la red con los datos del equipo.
    model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mae", "mse"])

    # EN: It is defined the callback of Early Stopping to avoid overfitting.
    # ES: Se define el callback de Early Stopping para evitar el sobreajuste.
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", # EN: Monitor the validation loss. ES: Monitorear la pérdida de validación.
        patience=10,         # EN: Number of epochs with no improvement after which training will be stopped.
                            # ES: Número de épocas sin mejora después de las cuales se detendrá el entrenamiento.
        restore_best_weights=True, # EN: Restore the best weights. ES: Restaurar los mejores pesos.
        mode="min"          # EN: The model will stop training when the quantity monitored has stopped decreasing.
                            # ES: El modelo dejará de entrenar cuando la cantidad monitoreada haya dejado de disminuir.
    )

    # EN: Get the size of the batch to be used in the network according to the size of the training dataset.
    # ES: Obtener el tamaño del lote a ser usado en la red de acuerdo al tamaño del conjunto de datos de entrenamiento.
    length_training_dataset = len(training_X)

    if length_training_dataset < 20000:
        batch_size = 32
    elif length_training_dataset < 50000:
        batch_size = 64
    elif length_training_dataset < 200000:
        batch_size = 128
    elif length_training_dataset < 500000:
        batch_size = 256
    elif length_training_dataset < 800000:
        batch_size = 512
    else:
        batch_size = 1024

    model.fit(training_X, training_y, validation_split=0.1,
              # EN: 10% of the training dataset will be used as validation dataset.
                # ES: El 10% del conjunto de datos de entrenamiento será usado como conjunto de datos de validación.
              batch_size=batch_size, epochs=epochs,
              callbacks=[early_stopping])

    # EN: Get the predictions of the network. ES: Obtener las predicciones de la red.
    predictions = model.predict(testing_X)

    # EN: Analyze the predictions of the network. ES: Analizar las predicciones de la red.
    comparison_predictions = analyze_predictions(predictions, testing_y)

    # EN: Set end time for the network. ES: Establecer el tiempo de finalización para la red.
    end_time = pd.Timestamp.now()

    # EN: Get the time that the network took to execute. ES: Obtener el tiempo que la red tardó en ejecutarse.
    time_execution = end_time - init_time

    # EN: Add new column to the dataframe with the time that the network took to execute.
    # ES: Agregar nueva columna al dataframe con el tiempo que la red tardó en ejecutarse.
    comparison_predictions.loc["Time Execution"] = time_execution

    return comparison_predictions


def incremental_rmse(testing_y, predictions, batch_size=1024):
    """
    list, list, int --> list
    OBJ: EN: Calculate the Root Mean Squared Error (RMSE) incrementally. ES: Calcular el Error Cuadrático Medio (RMSE)
    incrementalmente.
    :param testing_y: EN: Real values. ES: Valores reales.
    :param predictions: EN: Predictions of the network. ES: Predicciones de la red.
    :param batch_size: EN: Size of the batch. ES: Tamaño del lote.
    :return: list EN: List of the RMSE calculated incrementally. ES: Lista del RMSE calculado incrementalmente.
    """
    total_squared_error = 0
    total_samples = 0

    for i in range(0, len(testing_y), batch_size):
        batch_testing_y = testing_y[i:i + batch_size]
        batch_predictions = predictions[i:i + batch_size]

        batch_squared_error = np.sum(np.square(batch_testing_y - batch_predictions))
        total_squared_error += batch_squared_error
        total_samples += len(batch_testing_y)

    rmse = np.sqrt(total_squared_error / total_samples)
    return rmse


def analyze_predictions(predictions, testing_y):
    """
    list, list --> dataframe
    OBJ: EN: Check how good the predictions are from the network considering the real values. ES: Comprobar qué tan
    buenas son las predicciones de la red considerando los valores reales.
    :param predictions: EN: Predictions of the network. ES: Predicciones de la red.
    :param testing_y: EN: Real values. ES: Valores reales.
    :return: dataframe EN: Dataframe with the comparison results between the predictions and the real values.
    ES: Dataframe con los resultados de comparación entre las predicciones y los valores reales.
    """
    # EN: Set in a same dataframe the predictions and the real values.
    # ES: Establecer en un mismo dataframe las predicciones y los valores reales.
    df_predictions = pd.DataFrame(predictions, columns=["Predictions"])
    df_predictions["Real Values"] = testing_y

    # EN: Get the Explained Variance Regression Score.
    # ES: Obtener el Score de Regresión de Varianza Explicado.
    variance_score = explained_variance_score(testing_y, predictions)
    # EN: Get the Mean Squared Error (MSE).
    # ES: Obtener el Error Cuadrático Medio (MSE).
    mse = mean_squared_error(testing_y, predictions)
    # EN: Get the Root Mean Squared Error (RMSE).
    # ES: Obtener el Error Cuadrático Medio (RMSE).
    # EN: To get the value of the RMSE, it is necessary to convert the tensor to a numpy array.
    # ES: Para obtener el valor del RMSE, es necesario convertir el tensor a un array de numpy.
    #rmse = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(testing_y, predictions)))).numpy()
    rmse = incremental_rmse(testing_y, predictions)
    # EN: Get the Mean Squared Log Error (MSLE).
    # ES: Obtener el Error Logarítmico Cuadrático Medio (MSLE).
    msle = mean_squared_log_error(np.abs(testing_y), np.abs(predictions))
    # EN: Get the Mean Absolute Error (MAE).
    # ES: Obtener el Error Absoluto Medio (MAE).
    mae = mean_absolute_error(testing_y, predictions)
    # EN: Get the Median Absolute Error (MedAE).
    # ES: Obtener el Error Absoluto Mediano (MedAE).
    medae = median_absolute_error(testing_y, predictions)
    # EN: Get the Mean Absolute Percentage Error (MAPE).
    # ES: Obtener el Error Porcentual Absoluto Medio (MAPE).
    mape = mean_absolute_percentage_error(testing_y, predictions)
    # EN: Get the R2 Score.
    # ES: Obtener el Score R2.
    r2_score_value = r2_score(testing_y, predictions)
    # EN: Get the Mean Tweedie Deviance (MTD).
    # ES: Obtener la Desviación Tweedie Media (MTD).
    mtd = mean_tweedie_deviance(testing_y, predictions)
    # EN: Get the D2 Tweedie Score.
    # ES: Obtener el Score D2 Tweedie.
    d2_tweedie_score_value = d2_tweedie_score(testing_y, predictions)
    # EN: Get the Max Error.
    # ES: Obtener el Error Máximo.
    max_error_value = max_error(testing_y, predictions)

    # EN: Set the headers for the dataframe.
    # ES: Establecer los encabezados para el dataframe.
    headers = ["Explained Variance Score", "Mean Squared Error", "Root Mean Squared Error", "Mean Squared Log Error",
               "Mean Absolute Error", "Median Absolute Error", "Mean Absolute Percentage Error", "R2 Score",
               "Mean Tweedie Deviance", "D2 Tweedie Score", "Max Error"]
    # EN: Set the values for the dataframe.
    # ES: Establecer los valores para el dataframe.
    values = [variance_score, mse, rmse, msle, mae, medae, mape, r2_score_value, mtd, d2_tweedie_score_value,
                max_error_value]

    # EN: Create the dataframe with the comparison results between the predictions and the real values.
    # ES: Crear el dataframe con los resultados de comparación entre las predicciones y los valores reales.
    df_results = pd.DataFrame(values, index=headers, columns=["Values"])

    return df_results
