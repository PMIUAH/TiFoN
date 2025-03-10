import menu

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
