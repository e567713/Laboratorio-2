import utils

#########################
# Ejercicio del teórico #
#########################

# Data set del teórico
S = [
    {'Dedicacion': 'Alta', 'Dificultad': 'Alta', 'Horario': 'Nocturno',
        'Humedad': 'Media', 'Humor Docente': 'Bueno', 'Salva': 'Yes'},
    {'Dedicacion': 'Baja', 'Dificultad': 'Media', 'Horario': 'Matutino',
        'Humedad': 'Alta', 'Humor Docente': 'Malo', 'Salva': 'No'},
    {'Dedicacion': 'Media', 'Dificultad': 'Alta', 'Horario': 'Nocturno',
        'Humedad': 'Media', 'Humor Docente': 'Malo', 'Salva': 'Yes'},
    {'Dedicacion': 'Media', 'Dificultad': 'Alta', 'Horario': 'Matutino',
        'Humedad': 'Alta', 'Humor Docente': 'Bueno', 'Salva': 'No'},
]

S_entropy = utils.entropy(S, 'Salva')
print('Entropía del conjunto S: ', S_entropy)

S_information_gain = utils.information_gain(S, 'Dedicacion', 'Salva')
print('Information gain del atributo Dedicación: ', S_information_gain)
S_information_gain = utils.information_gain(S, 'Humor Docente', 'Salva')
print('Information gain del atributo Humor Docente: ', S_information_gain)
S_information_gain = utils.information_gain(S, 'Horario', 'Salva')
print('Information gain del atributo Horario: ', S_information_gain)

tree = utils.ID3_algorithm(
    S,
    ['Dedicacion', 'Dificultad', 'Horario', 'Humedad', 'Humor Docente'],
    'Salva')

utils.print_tree(tree, tree['data'], None, True, '')
#############################################
# Ejercicio con el data set del laboratorio #
#############################################

# Leemos data set del laboratorio
data_set = utils.read_file('Autism-Adult-Data.arff')
# Calculamos su entropía
data_set_entropy = utils.entropy(data_set, 'Class/ASD')
print(data_set_entropy)
