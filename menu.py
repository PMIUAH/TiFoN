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
