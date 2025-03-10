import menu
import networkStructure

def main():
    """
    OBJ: EN: Main function of the project. ES: Función principal del proyecto.
    :return:
    """
    verify = False
    option = 0
    while not verify:
        menu.show_menu()
        option = menu.select_option()
        verify = menu.check_option_selected(option)
        if not verify:
            print("The option selected is not valid.")

    # EN: Get the number of layers and the activation functions for each layer.
    # ES: Obtener el número de capas y las funciones de activación para cada capa.
    number_layers = menu.show_menu_selection_number_layers()
    activation_functions = menu.get_selected_functions_layers(number_layers)

    # EN: Get the number of neurons for each layer.
    # ES: Obtener el número de neuronas para cada capa.
    number_neurons_layers = menu.get_number_neurons_layers(number_layers)

    # EN: Set the input number shape for the model.
    # ES: Establecer la forma del número de entradas para el modelo.
    input_shape = 1
    if option == 1: # EN: Individual treatment of sensors. ES: Tratamiento individual de los sensores.
        input_shape = 47
    elif option == 2: # EN: Collective treatment of sensors. ES: Tratamiento colectivo de los sensores.
        input_shape = 48 # EN: 47 + 1 (column number of sensors). ES: 47 + 1 (column número de los sensores).


    # EN: Build the model with the selected options.
    # ES: Construir el modelo con las opciones seleccionadas.
    model = networkStructure.build_network_structure(activation_functions, number_neurons_layers,
                                                     input_shape=input_shape)

    # EN: Get the number of epochs for the training of the model.
    # ES: Obtener el número de épocas para el entrenamiento del modelo.
    number_epochs = menu.get_number_epochs()


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


main()
