import utils

# Data set del teórico
S = [
    {'Dedicacion': 'Alta', 'Dificultad': 'Alta', 'Horario': 'Nocturno',
        'Humedad': 'Media', 'Humor Doc': 'Bueno', 'Salva': 'Yes'},
    {'Dedicacion': 'Baja', 'Dificultad': 'Media', 'Horario': 'Matutino',
        'Humedad': 'Alta', 'Humor Doc': 'Malo', 'Salva': 'No'},
    {'Dedicacion': 'Media', 'Dificultad': 'Alta', 'Horario': 'Nocturno',
        'Humedad': 'Media', 'Humor Doc': 'Malo', 'Salva': 'Yes'},
    {'Dedicacion': 'Media', 'Dificultad': 'Alta', 'Horario': 'Matutino',
        'Humedad': 'Alta', 'Humor Doc': 'Bueno', 'Salva': 'No'},
]

# Leemos data set del laboratorio
data_set = utils.read_file('Autism-Adult-Data.arff')

# Calculamos la entropía de ambos conjuntos
S_entropy = utils.entropy(S, 'Salva')
data_set_entropy = utils.entropy(data_set, 'Class/ASD')

print(data_set_entropy)
print(S_entropy)

S_information_gain = utils.information_gain(S ,'Dedicacion')
print(S_information_gain)