import filesCSV as fcsv


def show_menu():
    """
    None --> None
    OBJ: EN: Show the options of the menu. ES: Muestra las opciones del menú.
    :return: None
    """
    print("Mark the option you want to execute by inserting the number of the option:")
    print("1. Make the window sliding prediction by individual sensors.")
    print("2. Make the window sliding prediction by collection of sensors.")


def select_option():
    """
    None --> int
    OBJ: EN: Select an option of the menu. ES: Seleccionar una opción del menú.
    :return: int
    """
    option = int(input("Insert the number of the option: "))
    return option


def check_option_selected(option):
    """
    int --> bool
    OBJ: EN: Check if the option selected is valid. ES: Comprobar si la opción seleccionada es válida.
    :param option: EN: Option selected. ES: Opción seleccionada.
    :return: bool
    """
    if 1 <= option <= 2:
        verify = True
    else:
        verify = False
    return verify


def show_menu_selection_number_layers():
    """
    None --> int
    OBJ: EN: Show the menu to select the number of layers and return the number of layers.
    ES: Muestra el menú para seleccionar el número de capas y devuelve el número de capas.
    :return: int
    """
    verify = False
    number_layers = 0
    while not verify:
        print("Introduce the number of layers for the model without counting the last layer:")
        # EN: It is necessary to add 1 to the number of layers to count the last layer.
        # ES: Es necesario sumar 1 al número de capas para contar la última capa.
        number_layers = int(input("Insert the number of layers: "))
        if number_layers > 0:
            verify = True
        else:
            print("The number of layers must be greater than 0.")

        # EN: As it is a predictive model, the last layer is counted as an additional layer to the number of layers
        # entered and will always have the identity/linear activation function.
        # ES: Al ser un modelo predictivo, la última capa se cuenta como una capa adicional al número de capas
        # introducido y siempre tendrá la función de activación identidad/lineal.
        number_layers += 1
    return number_layers


def show_menu_selection_activation_function(number_layer):
    """
    None --> str
    OBJ: EN: Show the menu to select the activation function for the layer and return the number of that function.
    ES: Muestra el menú para seleccionar la función de activación para la capa y devuelve el número de esa función.
    :param number_layer: EN: Number of the layer. ES: Número de la capa.
    :return: str
    """
    verify = False
    activation_function = 0
    while not verify:
        print("Mark the option you want to select for the activation function of the layer", number_layer, ":")
        print("1. Identity/Linear --> Used for prediction.")
        print("2. Binary Step --> Used for classification.")
        print("3. Sigmoid/Logistic --> Used for classification.")
        print("4. Hard Sigmoid --> Used for classification.")
        print("5. Elliot Sigmoid --> Used mainly for classification, but also for prediction in some occasions.")
        print("6. TanH/Hyperbolic Tangent --> Used for classification and prediction.")
        print("7. ReLU/Rectified Linear Unit --> Used for prediction, but also for classification in some occasions.")
        print("8. Leaky ReLU --> Used for prediction and classification.")
        print("9. RReLU/Randomized Leaky ReLU --> Used for prediction and classification.")
        print("10. PReLU/Parametric ReLU --> Used for prediction, but also for classification in some occasions.")
        print("11. GELU/Gaussian Error Linear Unit --> Used for prediction and classification.")
        print("12. SoftMax --> Used for classification.")
        print("13. SoftPlus --> Used for prediction, but also for classification in some occasions.")
        print("14. SoftSign --> Used for prediction and classification.")
        print("15. Maxout --> Used for prediction.")
        print("16. ELU/Exponential Linear Unit --> Used for prediction and classification.")
        print("17. SELU/Scaled Exponential Linear Unit --> Used for prediction and classification.")
        print("18. Swish --> Used for prediction and classification.")
        print("19. Mish --> Used for prediction and classification.")
        print("20. Bent Identity --> Used for prediction.")

        option = int(input("Insert the number of the option: "))
        if 1 <= option <= 20:
            verify = True
            activation_function = option
        else:
            print("The option selected is not valid.")
    return activation_function


def get_selected_functions_layers(number_layers):
    """
    int --> []
    OBJ: EN: Get the selected functions for the layers. ES: Obtener las funciones seleccionadas para las capas.
    :param number_layers: EN: Number of layers. ES: Número de capas.
    :return: []
    """
    functions_layers = []
    for i in range(1, number_layers + 1):
        # EN: As it is built a predictive model, the last layer will always have the identity/linear activation function.
        # ES: Al construir un modelo predictivo, la última capa siempre tendrá la función de activación identidad/lineal.
        if i == number_layers:
            functions_layers.append(1)
        else:
            functions_layers.append(show_menu_selection_activation_function(i))
    return functions_layers


def get_number_neurons_layers(number_layers):
    """
    int --> []
    OBJ: EN: Get the number of neurons for each layer. ES: Obtener el número de neuronas para cada capa.
    :param number_layers: EN: Number of layers. ES: Número de capas.
    :return: []
    """
    number_neurons_layers = []
    for i in range(1, number_layers + 1):
        # EN: As this predictive model is thought to be used for window sliding and predictive the last value of the
        # sequence, the last layer will always have only one neuron.
        # ES: Como este modelo predictivo está pensado para ser utilizado para el deslizamiento de ventanas y predecir
        # el último valor de la secuencia, la última capa siempre tendrá solo una neurona.
        if i == number_layers:
            number_neurons_layers.append(1)
        else:
            verify = False
            number_neurons = 0
            while not verify:
                print("Introduce the number of neurons for the layer", i, ":")
                number_neurons = int(input("Insert the number of neurons: "))
                if number_neurons > 0:
                    verify = True
                else:
                    print("The number of neurons must be greater than 0.")
            number_neurons_layers.append(number_neurons)
    return number_neurons_layers


def get_number_epochs():
    """
    None --> int
    OBJ: EN: Get the number of epochs for the training of the model. ES: Obtener el número de épocas para el entrenamiento
    del modelo.
    :return: int
    """
    verify = False
    number_epochs = 0
    while not verify:
        print("Introduce the number of epochs for the training of the model:")
        number_epochs = int(input("Insert the number of epochs: "))
        if number_epochs > 0:
            verify = True
        else:
            print("The number of epochs must be greater than 0.")
    return number_epochs


def select_all_sensors_or_set_sensors():
    """
    None --> list
    OBJ: EN: Show the menu to select if we want to analyze all the sensors individually or select a set of them.
    ES: Muestra el menú para seleccionar si queremos analizar todos los sensores individualmente o seleccionar un
    conjunto de ellos.
    :return: list EN: List with the selected sensors. ES: Lista con los sensores seleccionados.
    """
    verify = False
    option = 0
    sensors = []
    while not verify:
        print("Mark the option you want to select:")
        print("1. Analyze all the sensors individually.")
        print("2. Select a set of sensors.")

        option = int(input("Insert the number of the option: "))
        if 1 <= option <= 2:
            verify = True
        else:
            print("The option selected is not valid.")

    if option == 1:
        sensors_ids = fcsv.get_list_sensors_ids().values.transpose().tolist()[0]
        for i in range(len(sensors_ids)):
            sensors.append(sensors_ids[i])
    if option == 2:
        verify = False
        number_sensors = 0
        while not verify:
            print("Introduce the number of sensors you want to select:")
            number_sensors = int(input("Insert the number of sensors: "))
            if number_sensors > 0:
                verify = True
            else:
                print("The number of sensors must be greater than 0.")

        sensors_ids = fcsv.get_list_sensors_ids().values.transpose().tolist()[0]
        # EN: Show the list of sensors and select the sensors and check if the selected sensors are correct
        # and are not repeated. The list of sensors is shown in the same order as the IDs and printed in the console
        # to facilitate the selection of the sensors set 10 by 10.
        # ES: Mostrar la lista de sensores y seleccionar los sensores y comprobar si los sensores seleccionados son
        # correctos y no se repiten. La lista de sensores se muestra en el mismo orden que los IDs y se imprime en la
        # consola para facilitar la selección del conjunto de sensores de 10 en 10.
        print("List of sensors:")
        for i in range(len(sensors_ids)):
            print(sensors_ids[i], end=" ")
            if (i + 1) % 10 == 0:
                print()
        print()

        for i in range(number_sensors):
            verify = False
            while not verify:
                print("Introduce the ID of the sensor", i + 1, ":")
                equip_id = int(input("Insert the ID of the sensor: "))
                if equip_id in sensors_ids:
                    if equip_id not in sensors:
                        verify = True
                        sensors.append(equip_id)
                    else:
                        print("The sensor is already selected.")
                else:
                    print("The sensor ID is not valid.")

    return sensors
