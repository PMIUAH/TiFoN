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


