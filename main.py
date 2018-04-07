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

S2 = [{'Dedicacion': 'Alta', 'Dificultad': 'Alta', 'Horario': 'Matutino',
       'Humedad': 'Media', 'Humor Docente': 'Bueno', 'Salva': 'Yes'},
      {'Dedicacion': 'Baja', 'Dificultad': 'Media', 'Horario': 'Matutino',
       'Humedad': 'Alta', 'Humor Docente': 'Malo', 'Salva': 'No'},
      {'Dedicacion': 'Media', 'Dificultad': 'Alta', 'Horario': 'Nocturno',
       'Humedad': 'Media', 'Humor Docente': 'Malo', 'Salva': 'Yes'},
      {'Dedicacion': 'Media', 'Dificultad': 'Alta', 'Horario': 'Matutino',
       'Humedad': 'Media', 'Humor Docente': 'Bueno', 'Salva': 'No'}]

# S_entropy = utils.entropy(S, 'Salva')
# print('Entropía del conjunto S: ', S_entropy)

# S_information_gain = utils.information_gain(S2, 'Dedicacion', 'Salva')
# print('Information gain del atributo Dedicación: ', S_information_gain)
# S_information_gain = utils.information_gain(S2, 'Humor Docente', 'Salva')
# print('Information gain del atributo Humor Docente: ', S_information_gain)
# S_information_gain = utils.information_gain(S2, 'Horario', 'Salva')
# print('Information gain del atributo Horario: ', S_information_gain)

# tree = utils.ID3_algorithm(
#     S,
#     ['Dedicacion', 'Dificultad', 'Horario', 'Humedad', 'Humor Docente'],
#     'Salva')

# utils.print_tree(tree, tree['data'], None, True, '')

# tree = utils.ID3_algorithm(
#     S2,
#     ['Dedicacion', 'Dificultad', 'Horario', 'Humedad', 'Humor Docente'],
#     'Salva')

# utils.print_tree(tree, tree['data'], None, True, '')
#############################################
# Ejercicio con el data set del laboratorio #
#############################################

# Leemos data set del laboratorio
examples = utils.read_file('Autism-Adult-Data.arff')
data_set = examples[0]
metadata = examples[1]
# Calculamos su entropía.
# data_set_entropy = utils.entropy(data_set, 'Class/ASD')
# print(data_set_entropy)

# print(type(metadata))

# tree_2 = utils.ID3_algorithm(
#     data_set,
#     ['A1_Score',
#      'A2_Score',
#      'A3_Score',
#      'A4_Score',
#      'A5_Score',
#      'A6_Score',
#      'A7_Score',
#      'A8_Score',
#      'A9_Score',
#      'A10_Score',
#      'age',
#      'gender',
#      'ethnicity',
#      'jundice',
#      'austim',
#      'contry_of_res',
#      'used_app_before',
#      'result',
#      'age_desc',
#      'relation'],
#     'Class/ASD')

tree_2 = utils.ID3_algorithm_with_threshold(
    data_set,
    ['A1_Score',
     'A2_Score',
     'A3_Score',
     'A4_Score',
     'A5_Score',
     'A6_Score',
     'A7_Score',
     'A8_Score',
     'A9_Score',
     'A10_Score',
     'age',
     'gender',
     'ethnicity',
     'jundice',
     'austim',
     'contry_of_res',
     'used_app_before',
     # 'result',
     'age_desc',
     'relation'],
    'Class/ASD',
    ['age',
     'result'])

# utils.print_tree(tree_2, tree_2['data'], None, True, '')


def menu():
    clear()
    print('Seleccione una opción:')
    print('   1 - (Ejercicio 5a)')
    print('   2 - (Ejercicio 5b)')
    print('   3 - Salir')
    choise = input('Ingrese opción:  ')
    if choise == '1':
        optionOne()
    elif choise == '2':
        repetition = int(input('Ingrese número de repeticiones para calcular promedio: '))
        optionTwo(repetition)
Correr algoritmo ID3 con el ejemplo del teórico

def optionOne():
    clear()
    print('Tabla del Teórico')
    print('-----------------')
    table.print()
    print('\n')
    print('La hipótesis es:')
    print('---------------')
    print(findS.calculateFind_S(table))
    print('\n')
    input("'Enter' para volver a menu:  ")
    menu()

def optionTwo(repetition):
    clear()
    print('Promedio de instancias únicas necesarias para aprender el concepto: ')
    print(allInstancesFindS(repetition))
    print('\n')
    print('Promedio de instancias únicas positivas necesarias para aprender el concepto: ')
    print(positiveInstancesFindS(repetition))
    print('\n')
    input("'Enter' para volver a menu:  ")
    menu()