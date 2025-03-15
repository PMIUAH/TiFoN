import pandas as pd
import tensorflow as tf


def build_network_structure(list_activation_functions_layers, list_number_neurons_layers, input_shape=47):
    """
    list, list, list, int --> model
    OBJ: EN: Build the network structure with the activation functions and the number of neurons for each layer.
    ES: Construir la estructura de la red con las funciones de activación y el número de neuronas para cada capa.
    :param list_activation_functions_layers: EN: List of activation functions for each layer. ES: Lista de funciones de
    activación para cada capa.
    :param list_number_neurons_layers: EN: List of the number of neurons for each layer. ES: Lista del número de
    neuronas para cada capa.
    :param input_shape: EN: Number of inputs. ES: Número de entradas.
    :return: EN: Model of the network. ES: Modelo de la red.
    """
    # EN: Create the model with the selected options.
    # ES: Crear el modelo con las opciones seleccionadas.
    model = tf.keras.models.Sequential()

    # EN: Add the input layer to the model indicating the input shape.
    # ES: Añadir la capa de entrada al modelo indicando la forma de entrada.
    model.add(tf.keras.layers.InputLayer(shape=(input_shape,)))

    # EN: Add the layers to the model.
    # ES: Añadir las capas al modelo.
    for i in range(len(list_activation_functions_layers)):
        model.add(build_network_structure_layer(list_activation_functions_layers[i], list_number_neurons_layers[i]))
        print("Layer", i + 1, "added.")

    return model


def build_model_network_dataframe(list_activation_functions_layers, list_number_neurons_layers, number_epochs):
    """
    list, list, int --> dataframe
    OBJ: EN: Build the model of the network as a dataframe structure.
    ES: Construir el modelo de la red como una estructura de dataframe.
    :param list_activation_functions_layers: EN: List of activation functions for each layer. ES: Lista de funciones de
    activación para cada capa.
    :param list_number_neurons_layers: EN: List of the number of neurons for each layer. ES: Lista del número de
    neuronas para cada capa.
    :param number_epochs: EN: Number of epochs. ES: Número de épocas.
    :return:
    """
    headers = ["Layer", "Activation Function", "Number of Neurons", "Number of Epochs"]
    model_list = []
    for i in range(len(list_activation_functions_layers)):
        model_list.append(
            [
                i + 1,
                identify_activation_function(list_activation_functions_layers[i]),
                list_number_neurons_layers[i],
                number_epochs]
        )
    model_dataframe = pd.DataFrame(model_list, columns=headers)

    return model_dataframe


def build_network_structure_layer(number_activation_function, number_neurons_layers):
    """
    int, int, int, bool --> layer
    OBJ: EN: Build the network structure layer with the activation function and the number of neurons.
    ES: Construir la capa de estructura de red con la función de activación y el número de neuronas.
    :param number_activation_function: EN: Number of the activation function. ES: Número de la función de activación.
    :param number_neurons_layers: EN: Number of neurons. ES: Número de neuronas.
    :return: EN: Model of the layer. ES: Modelo de la capa.
    """
    layer = None

    # EN: Identity/Linear activation function.
    # ES: Función de activación identidad/lineal.
    if number_activation_function == 1:
        layer = build_layer_model_identity_activation_function(number_neurons_layers)
    # EN: Step activation function.
    # ES: Función de activación paso.
    elif number_activation_function == 2:
        layer = build_layer_model_step_activation_function(number_neurons_layers)
    # EN: Sigmoid activation function.
    # ES: Función de activación sigmoid.
    elif number_activation_function == 3:
        layer = build_layer_model_sigmoid_activation_function(number_neurons_layers)
    # EN: Hard sigmoid activation function.
    # ES: Función de activación hard sigmoid.
    elif number_activation_function == 4:
        layer = build_layer_model_hard_sigmoid_activation_function(number_neurons_layers)
    # EN: Elliot sigmoid activation function.
    # ES: Función de activación Elliot sigmoid.
    elif number_activation_function == 5:
        layer = build_layer_model_elliot_sigmoid_activation_function(number_neurons_layers)
    # EN: TanH activation function.
    # ES: Función de activación tanh.
    elif number_activation_function == 6:
        layer = build_layer_model_tanh_activation_function(number_neurons_layers)
    # EN: ReLU activation function.
    # ES: Función de activación ReLU.
    elif number_activation_function == 7:
        layer = build_layer_model_relu_activation_function(number_neurons_layers)
    # EN: Leaky ReLU activation function. This function does not need to specify the number of neurons, as it adjusted
    # internally.
    # ES: Función de activación Leaky ReLU. Esta función no necesita especificar el número de neuronas, ya que se ajusta
    # internamente.
    elif number_activation_function == 8:
        # EN: This activation will work only with the Dense layer. ES: Esta activación solo funcionará con la capa densa.
        layer = build_layer_model_leaky_relu_activation_function()
    # EN: RReLU activation function.
    # ES: Función de activación RReLU.
    elif number_activation_function == 9:
        layer = build_layer_model_rrelu_activation_function(number_neurons_layers)
    # EN: PReLU activation function. This function does not need to specify the number of neurons, as it adjusted
    # internally.
    # ES: Función de activación PReLU Esta función no necesita especificar el número de neuronas, ya que se ajusta
    # internamente.
    elif number_activation_function == 10:
        # EN: This activation will work only with the Dense layer. ES: Esta activación solo funcionará con la capa densa.
        layer = build_layer_model_prelu_activation_function()
    # EN: GELU activation function.
    # ES: Función de activación GELU.
    elif number_activation_function == 11:
        layer = build_layer_model_gelu_activation_function(number_neurons_layers)
    # EN: SoftMax activation function.
    # ES: Función de activación softmax
    elif number_activation_function == 12:
        layer = build_layer_model_softmax_activation_function(number_neurons_layers)
    # EN: SoftPlus activation function.
    # ES: Función de activación softplus.
    elif number_activation_function == 13:
        layer = build_layer_model_softplus_activation_function(number_neurons_layers)
    # EN: SoftSign activation function.
    # ES: Función de activación softsign.
    elif number_activation_function == 14:
        layer = build_layer_model_softsign_activation_function(number_neurons_layers)
    # EN: Maxout activation function.
    # ES: Función de activación maxout.
    elif number_activation_function == 15:
        # EN: This activation will work only with the Dense layer. ES: Esta activación solo funcionará con la capa densa.
        layer = build_layer_model_maxout_activation_function(number_neurons_layers)
    # EN: ELU activation function.
    # ES: Función de activación ELU.
    elif number_activation_function == 16:
        layer = build_layer_model_elu_activation_function(number_neurons_layers)
    # EN: SELU activation function.
    # ES: Función de activación SELU.
    elif number_activation_function == 17:
        layer = build_layer_model_selu_activation_function(number_neurons_layers)
    # EN: Swish activation function.
    # ES: Función de activación Swish.
    elif number_activation_function == 18:
        layer = build_layer_model_swish_activation_function(number_neurons_layers)
    # EN: Mish activation function.
    # ES: Función de activación Mish.
    elif number_activation_function == 19:
        layer = build_layer_model_mish_activation_function(number_neurons_layers)
    # EN: Bent Identity activation function.
    # ES: Función de activación Bent Identity.
    elif number_activation_function == 20:
        layer = build_layer_model_bent_identity_activation_function(number_neurons_layers)

    return layer


def identify_activation_function(number_activation_function):
    """
    int --> String
    OBJ: EN: Identify the activation function. ES: Identificar la función de activación.
    :param number_activation_function: EN: Number of the activation function. ES: Número de la función de activación.
    :return: EN: Activation function. ES: Función de activación.
    """
    name_activation_function = ""
    # EN: Identity/Linear activation function.
    # ES: Función de activación identidad/lineal.
    if number_activation_function == 1:
        name_activation_function = "Identity/Linear"
    # EN: Step activation function.
    # ES: Función de activación paso.
    elif number_activation_function == 2:
        name_activation_function = "Binary Step"
    # EN: Sigmoid activation function.
    # ES: Función de activación sigmoid.
    elif number_activation_function == 3:
        name_activation_function = "Sigmoid/Logistic"
    # EN: Hard sigmoid activation function.
    # ES: Función de activación hard sigmoid.
    elif number_activation_function == 4:
        name_activation_function = "Hard Sigmoid"
    # EN: Elliot sigmoid activation function.
    # ES: Función de activación Elliot sigmoid.
    elif number_activation_function == 5:
        name_activation_function = "Elliot Sigmoid"
    # EN: TanH activation function.
    # ES: Función de activación tanh.
    elif number_activation_function == 6:
        name_activation_function = "TanH"
    # EN: ReLU activation function.
    # ES: Función de activación ReLU.
    elif number_activation_function == 7:
        name_activation_function = "ReLU"
    # EN: Leaky ReLU activation function.
    # ES: Función de activación Leaky ReLU.
    elif number_activation_function == 8:
        name_activation_function = "Leaky ReLU"
    # EN: RReLU activation function.
    # ES: Función de activación RReLU.
    elif number_activation_function == 9:
        name_activation_function = "RReLU"
    # EN: PReLU activation function.
    # ES: Función de activación PReLU.
    elif number_activation_function == 10:
        name_activation_function = "PReLU"
    # EN: GELU activation function.
    # ES: Función de activación GELU.
    elif number_activation_function == 11:
        name_activation_function = "GELU"
    # EN: SoftMax activation function.
    # ES: Función de activación softmax
    elif number_activation_function == 12:
        name_activation_function = "SoftMax"
    # EN: SoftPlus activation function.
    # ES: Función de activación softplus.
    elif number_activation_function == 13:
        name_activation_function = "SoftPlus"
    # EN: SoftSign activation function.
    # ES: Función de activación softsign.
    elif number_activation_function == 14:
        name_activation_function = "SoftSign"
    # EN: Maxout activation function.
    # ES: Función de activación maxout.
    elif number_activation_function == 15:
        name_activation_function = "Maxout"
    # EN: ELU activation function.
    # ES: Función de activación ELU.
    elif number_activation_function == 16:
        name_activation_function = "ELU"
    # EN: SELU activation function.
    # ES: Función de activación SELU.
    elif number_activation_function == 17:
        name_activation_function = "SELU"
    # EN: Swish activation function.
    # ES: Función de activación Swish.
    elif number_activation_function == 18:
        name_activation_function = "Swish"
    # EN: Mish activation function.
    # ES: Función de activación Mish.
    elif number_activation_function == 19:
        name_activation_function = "Mish"
    # EN: Bent Identity activation function.
    # ES: Función de activación Bent Identity.
    elif number_activation_function == 20:
        name_activation_function = "Bent Identity"

    return name_activation_function


def build_layer_model_identity_activation_function(number_neurons):
    """
    int, int, bool --> layer
    OBJ: EN: Build the layer model with the activation function identity.
    ES: Construir el modelo de capa con la función de activación identidad.
    :param number_neurons: EN: Number of neurons. ES: Número de neuronas.
    :param option_layer: EN: Option of the layer. ES: Opción de la capa.
    ES: Verificar si el tipo de capa se repetirá en la siguiente capa.
    :return: EN: Model of the layer. ES: Modelo de la capa.
    """
    # EN: Build the layer model with the activation function identity.
    # ES: Construir el modelo de capa con la función de activación identidad.
    layer = tf.keras.layers.Dense(number_neurons, activation='linear')

    return layer


def step_function(x):
    """
    float --> float
    OBJ: EN: Given a value, return 1 if it is greater than 0, otherwise return 0. ES: Dado un valor, devolver 1 si es
    mayor que 0, de lo contrario devolver 0.
    :param x: float EN: Value. ES: Valor.
    :return: float EN: 1 if the value is greater than 0, otherwise 0.
    ES: 1 si el valor es mayor que 0, de lo contrario 0.
    """
    return tf.cast(x > 0, dtype=tf.float32)


def build_layer_model_step_activation_function(number_neurons):
    """
    int, int, bool --> layer
    OBJ: EN: Build the layer model with the activation function step.
    ES: Construir el modelo de capa con la función de activación paso.
    :param number_neurons: EN: Number of neurons. ES: Número de neuronas.
    :return: EN: Model of the layer. ES: Modelo de la capa.
    """
    # EN: Build the layer model with the activation function step.
    # ES: Construir el modelo de capa con la función de activación paso.
    layer = tf.keras.layers.Dense(number_neurons, activation=step_function)

    return layer


def build_layer_model_sigmoid_activation_function(number_neurons):
    """
    int, int, bool --> layer
    OBJ: EN: Build the layer model with the activation function sigmoid.
    ES: Construir el modelo de capa con la función de activación sigmoid.
    :param number_neurons: EN: Number of neurons. ES: Número de neuronas.
    :return: EN: Model of the layer. ES: Modelo de la capa.
    """
    # EN: Build the layer model with the activation function sigmoid.
    # ES: Construir el modelo de capa con la función de activación sigmoid.
    layer = tf.keras.layers.Dense(number_neurons, activation='sigmoid')

    return layer


def build_layer_model_hard_sigmoid_activation_function(number_neurons):
    """
    int, int, bool --> layer
    OBJ: EN: Build the layer model with the activation function hard sigmoid.
    ES: Construir el modelo de capa con la función de activación hard sigmoid.
    :param number_neurons: EN: Number of neurons. ES: Número de neuronas.
    :return: EN: Model of the layer. ES: Modelo de la capa.
    """
    # EN: Build the layer model with the activation function hard sigmoid.
    # ES: Construir el modelo de capa con la función de activación hard sigmoid.
    layer = tf.keras.layers.Dense(number_neurons, activation='hard_sigmoid')

    return layer


def elliot_sigmoid(x):
    """
    float --> float
    OBJ: EN: Given a value, return the value of the Elliot sigmoid function. ES: Dado un valor, devolver el valor de la
    función de Elliot sigmoid.
    :param x: float EN: Value. ES: Valor.
    :return: float EN: Value of the Elliot sigmoid function. ES: Valor de la función de Elliot sigmoid.
    """
    return x / (1 + abs(x))


def build_layer_model_elliot_sigmoid_activation_function(number_neurons):
    """
    int, int, bool --> layer
    OBJ: EN: Build the layer model with the activation function Elliot sigmoid.
    ES: Construir el modelo de capa con la función de activación Elliot sigmoid.
    :param number_neurons: EN: Number of neurons. ES: Número de neuronas.
    :return: EN: Model of the layer. ES: Modelo de la capa.
    """
    # EN: Build the layer model with the activation function Elliot sigmoid.
    # ES: Construir el modelo de capa con la función de activación Elliot sigmoid.
    layer = tf.keras.layers.Dense(number_neurons, activation=elliot_sigmoid)

    return layer


def build_layer_model_tanh_activation_function(number_neurons):
    """
    int, int, bool --> layer
    OBJ: EN: Build the layer model with the activation function tanh.
    ES: Construir el modelo de capa con la función de activación tanh.
    :param number_neurons: EN: Number of neurons. ES: Número de neuronas.
    :return: EN: Model of the layer. ES: Modelo de la capa.
    """
    # EN: Build the layer model with the activation function tanh.
    # ES: Construir el modelo de capa con la función de activación tanh.
    layer = tf.keras.layers.Dense(number_neurons, activation='tanh')

    return layer


def build_layer_model_relu_activation_function(number_neurons):
    """
    int, int, bool --> layer
    OBJ: EN: Build the layer model with the activation function ReLU.
    ES: Construir el modelo de capa con la función de activación ReLU.
    :param number_neurons: EN: Number of neurons. ES: Número de neuronas.
    :return: EN: Model of the layer. ES: Modelo de la capa.
    """
    # EN: Build the layer model with the activation function ReLU.
    # ES: Construir el modelo de capa con la función de activación ReLU.
    layer = tf.keras.layers.Dense(number_neurons, activation='relu')

    return layer


def build_layer_model_leaky_relu_activation_function():
    """
    --> layer
    OBJ: EN: Build the layer model with the activation function Leaky ReLU.
    ES: Construir el modelo de capa con la función de activación Leaky ReLU.
    :return: EN: Model of the layer. ES: Modelo de la capa.
    """
    # EN: Build the layer model with the activation function Leaky ReLU.
    # ES: Construir el modelo de capa con la función de activación Leaky ReLU.
    layer = tf.keras.layers.LeakyReLU(negative_slope=0.01)
    return layer


def rrelu(x, lower=0.01, upper=0.1):
    """
    float, float, float --> float
    OBJ: EN: Given a value, return the value of the RReLU function. ES: Dado un valor, devolver el valor de la función de
    RReLU.
    :param x: float EN: Value. ES: Valor.
    :param lower: float EN: Lower limit. ES: Límite inferior.
    :param upper: float EN: Upper limit. ES: Límite superior.
    :return: float EN: Value of the RReLU function. ES: Valor de la función de RReLU.
    """
    # EN: Random number between lower and upper.
    # ES: Número aleatorio entre lower y upper.
    alpha = tf.random.uniform(shape=tf.shape(x), minval=lower, maxval=upper)

    # EN: Apply the RReLU function.
    # ES: Aplicar la función de RReLU.
    return tf.where(x >= 0, x, alpha * x)


def build_layer_model_rrelu_activation_function(number_neurons):
    """
    int, int, bool --> layer
    OBJ: EN: Build the layer model with the activation function RReLU.
    ES: Construir el modelo de capa con la función de activación RReLU.
    :param number_neurons: EN: Number of neurons. ES: Número de neuronas.
    :return: EN: Model of the layer. ES: Modelo de la capa.
    """
    # EN: Build the layer model with the activation function RReLU.
    # ES: Construir el modelo de capa con la función de activación RReLU.
    layer = tf.keras.layers.Dense(number_neurons, activation=rrelu)

    return layer


def build_layer_model_prelu_activation_function():
    """
    --> layer
    OBJ: EN: Build the layer model with the activation function PReLU.
    ES: Construir el modelo de capa con la función de activación PReLU.
    :return: EN: Model of the layer. ES: Modelo de la capa.
    """
    # EN: Build the layer model with the activation function PReLU.
    # ES: Construir el modelo de capa con la función de activación PReLU.
    layer = tf.keras.layers.PReLU()
    return layer


def build_layer_model_gelu_activation_function(number_neurons):
    """
    int, int, bool --> layer
    OBJ: EN: Build the layer model with the activation function GELU.
    ES: Construir el modelo de capa con la función de activación GELU.
    :param number_neurons: EN: Number of neurons. ES: Número de neuronas.
    :return: EN: Model of the layer. ES: Modelo de la capa.
    """
    # EN: Build the layer model with the activation function GELU.
    # ES: Construir el modelo de capa con la función de activación GELU.
    layer = tf.keras.layers.Dense(number_neurons, activation='gelu')

    return layer


def build_layer_model_softmax_activation_function(number_neurons):
    """
    int, int, bool --> layer
    OBJ: EN: Build the layer model with the activation function softmax.
    ES: Construir el modelo de capa con la función de activación softmax.
    :param number_neurons: EN: Number of neurons. ES: Número de neuronas.
    :return: EN: Model of the layer. ES: Modelo de la capa.
    """
    # EN: Build the layer model with the activation function softmax.
    # ES: Construir el modelo de capa con la función de activación softmax.
    layer = tf.keras.layers.Dense(number_neurons, activation='softmax')

    return layer


def build_layer_model_softplus_activation_function(number_neurons):
    """
    int, int, bool --> layer
    OBJ: EN: Build the layer model with the activation function softplus.
    ES: Construir el modelo de capa con la función de activación softplus.
    :param number_neurons: EN: Number of neurons. ES: Número de neuronas.
    :return: EN: Model of the layer. ES: Modelo de la capa.
    """
    # EN: Build the layer model with the activation function softplus.
    # ES: Construir el modelo de capa con la función de activación softplus.
    layer = tf.keras.layers.Dense(number_neurons, activation='softplus')

    return layer


def build_layer_model_softsign_activation_function(number_neurons):
    """
    int, int, bool --> layer
    OBJ: EN: Build the layer model with the activation function softsign.
    ES: Construir el modelo de capa con la función de activación softsign.
    :param number_neurons: EN: Number of neurons. ES: Número de neuronas.
    :return: EN: Model of the layer. ES: Modelo de la capa.
    """
    # EN: Build the layer model with the activation function softsign.
    # ES: Construir el modelo de capa con la función de activación softsign.
    layer = tf.keras.layers.Dense(number_neurons, activation='softsign')

    return layer


def maxout(x, number_neurons):
    """
    float, int --> float
    OBJ: EN: Given a value and the number of neurons, return the value of the maxout function. ES: Dado un valor y el
    número de neuronas, devolver el valor de la función de maxout.
    :param x: float EN: Value. ES: Valor.
    :param number_neurons: int EN: Number of neurons. ES: Número de neuronas.
    :return: float EN: Value of the maxout function. ES: Valor de la función de maxout.
    """
    shape = tf.shape(x)
    x = tf.reshape(x, [-1, number_neurons, shape[-1] // number_neurons])
    return tf.reduce_max(x, axis=1)


def build_layer_model_maxout_activation_function(number_neurons):
    """
    int --> layer
    OBJ: EN: Build the layer model with the activation function maxout.
    ES: Construir el modelo de capa con la función de activación maxout.
    :param number_neurons: EN: Number of neurons. ES: Número de neuronas.
    :return: EN: Model of the layer. ES: Modelo de la capa.
    """
    # EN: Build the layer model with the activation function maxout.
    # ES: Construir el modelo de capa con la función de activación maxout.
    layer = tf.keras.layers.Lambda(lambda x: maxout(x, number_neurons))
    return layer


def build_layer_model_elu_activation_function(number_neurons):
    """
    int, int, bool --> layer
    OBJ: EN: Build the layer model with the activation function ELU.
    ES: Construir el modelo de capa con la función de activación ELU.
    :param number_neurons: EN: Number of neurons. ES: Número de neuronas.
    :return: EN: Model of the layer. ES: Modelo de la capa.
    """
    # EN: Build the layer model with the activation function ELU.
    # ES: Construir el modelo de capa con la función de activación ELU.
    layer = tf.keras.layers.Dense(number_neurons, activation='elu')

    return layer


def build_layer_model_selu_activation_function(number_neurons):
    """
    int, int, bool --> layer
    OBJ: EN: Build the layer model with the activation function SELU.
    ES: Construir el modelo de capa con la función de activación SELU.
    :param number_neurons: EN: Number of neurons. ES: Número de neuronas.
    :return: EN: Model of the layer. ES: Modelo de la capa.
    """
    # EN: Build the layer model with the activation function SELU.
    # ES: Construir el modelo de capa con la función de activación SELU.
    layer = tf.keras.layers.Dense(number_neurons, activation='selu')

    return layer


def build_layer_model_swish_activation_function(number_neurons):
    """
    int, int, bool --> layer
    OBJ: EN: Build the layer model with the activation function Swish.
    ES: Construir el modelo de capa con la función de activación Swish.
    :param number_neurons: EN: Number of neurons. ES: Número de neuronas.
    :return: EN: Model of the layer. ES: Modelo de la capa.
    """
    # EN: Build the layer model with the activation function Swish.
    # ES: Construir el modelo de capa con la función de activación Swish.
    layer = tf.keras.layers.Dense(number_neurons, activation='swish')

    return layer


def mish(x):
    """
    float --> float
    OBJ: EN: Given a value, return the value of the Mish function. ES: Dado un valor, devolver el valor de la función de
    Mish.
    :param x: float EN: Value. ES: Valor.
    :return: float EN: Value of the Mish function. ES: Valor de la función de Mish.
    """
    return x * tf.math.tanh(tf.math.softplus(x))


def build_layer_model_mish_activation_function(number_neurons):
    """
    int, int, bool --> layer
    OBJ: EN: Build the layer model with the activation function Mish.
    ES: Construir el modelo de capa con la función de activación Mish.
    :param number_neurons: EN: Number of neurons. ES: Número de neuronas.
    :return: EN: Model of the layer. ES: Modelo de la capa.
    """
    # EN: Build the layer model with the activation function Mish.
    # ES: Construir el modelo de capa con la función de activación Mish.
    layer = tf.keras.layers.Dense(number_neurons, activation='mish')

    return layer


def bent_identity(x):
    """
    float --> float
    OBJ: EN: Given a value, return the value of the Bent Identity function. ES: Dado un valor, devolver el valor de la
    función de Bent Identity.
    :param x: float EN: Value. ES: Valor.
    :return: float EN: Value of the Bent Identity function. ES: Valor de la función de Bent Identity.
    """
    return (tf.math.sqrt(tf.math.pow(x, 2) + 1) - 1) / 2 + x


def build_layer_model_bent_identity_activation_function(number_neurons):
    """
    int, int, bool --> layer
    OBJ: EN: Build the layer model with the activation function Bent Identity.
    ES: Construir el modelo de capa con la función de activación Bent Identity.
    :param number_neurons: EN: Number of neurons. ES: Número de neuronas.
    :return: EN: Model of the layer. ES: Modelo de la capa.
    """
    # EN: Build the layer model with the activation function Bent Identity.
    # ES: Construir el modelo de capa con la función de activación Bent Identity.
    layer = tf.keras.layers.Dense(number_neurons, activation=bent_identity)

    return layer
