import math
from scipy.io import arff
from collections import Counter
import copy
import random


def entropy(data, target_attr):
    # Calcula la entropía del conjunto data dado para el atributo target_attr.

    frequencies = {}
    data_entropy = 0.0

    # Calcula la frecuencia de cada valor en el atributo objetivo.
    for instance in data:
        if (instance[target_attr] in frequencies):
            frequencies[instance[target_attr]] += 1.0
        else:
            frequencies[instance[target_attr]] = 1.0

    # Para cada valor del atributo objetivo se calcula su proporción
    # dentro del conjunto data y se aplica la fórmula de entropía.
    for frequency in frequencies.values():
        data_entropy -= (frequency / len(data)) * \
            math.log(frequency / len(data), 2)

    return data_entropy


def information_gain(data, attr, target_attr):
    # Calcula la ganancia de información del atributo attr sobre el
    # conjunto data.

    data_subsets = {}
    data_information_gain = 0.0

    # Se divide el conjunto data en subconjuntos que tienen en común
    # el valor del atributo attr.
    for instance in data:
        if (instance[attr] in data_subsets):
            data_subsets[instance[attr]].append(instance)
        else:
            data_subsets[instance[attr]] = [instance]

    # Extension para valores faltantes primer metodo
    if '-' in data_subsets:
        common_value = find_most_common_value_in_S(data_subsets, target_attr)
        data_subsets[common_value].extend(data_subsets['-'])
        data_subsets.pop('-', None)


    # Se calcula el valor de information gain según lo visto en teórico.
    data_information_gain = entropy(data, target_attr)
    for data_subset in data_subsets.values():
        data_information_gain -= (len(data_subset) / len(data)) * \
            entropy(data_subset, target_attr)
    return data_information_gain


def ID3_algorithm(data, attributes, target_attr):
    # Voy poniendo los links de donde saco las cosas por si viaja algo fijarme

    # Genera lista únicamente con los valores del target attribute.
    #   https://stackoverflow.com/questions/25050311/extract-first-item-of-each-sublist-in-python
    target_attr_values = [instance[target_attr] for instance in data]

    # Si todas las instancias tienen el mismo valor → etiquetar con ese valor.
    #   https://stackoverflow.com/questions/3844801/check-if-all-elements-in-a-list-are-identical
    if (target_attr_values[1:] == target_attr_values[:-1]):
        return {
            'data': target_attr_values[0],
            'childs': {}
        }

    # Si no me quedan atributos → etiquetar con el valor más común.
    elif not attributes:
        return {
            'data': most_common(target_attr_values),
            'childs': {}
        }

    # En caso contrario
    else:
        # Se obtiene el atributo best_attr que mejor clasifica los ejemplos.
        best_attr = get_best_attribute(data, attributes, target_attr)

        # Se obtienen los valores que puede tomar el atributo best_attr.
        best_attr_values = set(instance[best_attr] for instance in data)

        # Se genera un árbol con el atributo best_attr en su raíz.
        tree = {
            'data': best_attr,
            'childs': {}
        }
        for value in best_attr_values:

            # Nos quedamos con los ejemplos de data que tengan el valor
            # value para el atributo best_attribute.
            filtered_data = [
                instance for instance in data if instance[best_attr] == value]

            # Si filtered_data es vacío → etiquetar con el valor más probable.
            if not filtered_data:
                tree['childs'][value] = {
                    'data': most_common(target_attr_values),
                    'childs': {}
                }

            # Si filtered_data no es vacío
            else:
                # Se quita a best_attr de la lista de atributos.
                filtered_attributes = copy.deepcopy(attributes)
                filtered_attributes.remove(best_attr)
                tree['childs'][value] = ID3_algorithm(
                    filtered_data, filtered_attributes, target_attr)
        return tree


def read_file(path):
    return arff.loadarff(path)


def most_common(lst):
    # Retorna el elemento más común dentro de la lista pasada por parámetro.
    #   https://stackoverflow.com/questions/1518522/python-most-common-element-in-a-list
    data = Counter(lst)
    return max(lst, key=data.get)


def get_best_attribute(data, attributes, target_attr):
    # Elige el mejor atributo medido según la ganancia de información que brinda.
    # Si existe más de un atributo con ganancia máxima para las condiciones dadas,
    # se devuelve uno aleatorio entre ellos.
    maximum_values_tied = []
    max_ig = -1
    for attr in attributes:
        ig = information_gain(data, attr, target_attr)
        if ig > max_ig:
            max_ig = ig
            maximum_values_tied = []
            maximum_values_tied.append(attr)
        elif ig == max_ig:
            maximum_values_tied.append(attr)
    return random.choice(maximum_values_tied)


def print_tree(tree, attr, childIsSheet, first, tab):
    print(tab + attr)
    if not childIsSheet and not first:
        tab = '   ' + tab[:len(tab) - 3] + '  |--'
        print(tab + tree['data'])
    if tree['childs']:
        for key, value in tree['childs'].items():
            if value['childs']:
                print_tree(value, key, False, False, '   ' +
                           tab[:len(tab) - 3] + '  |--')
            else:
                print_tree(value, key, True, False, '   ' +
                           tab[:len(tab) - 3] + '  |--')
    else:
        print('   ' + tab[:len(tab) - 3] + '  |--' + tree['data'])


def ID3_algorithm_with_threshold(data, attributes, target_attr, numeric_attributes):
    # Algoritmo ID3 extendido que divide el rango de valores posibles de los atributos
    # numéricos en dos. (Utiliza un único threshold para cada atributo)

    # Se cambian todos los valores de los atributos de tipo numeric
    splitted_data = split_numeric_attributes(
        copy.deepcopy(data), target_attr, numeric_attributes)

    # Se llama a la función ID3 ya implementada
    return ID3_algorithm(splitted_data, attributes, target_attr)


def split_numeric_attributes(data, target_attr, numeric_attributes):
    # Recorre los atributos numéricos cambiando sus posibles valores según 
    # un salto calclulado.
    for numeric_attr in numeric_attributes:
        split_by_best_threshold(data, target_attr, numeric_attr)

    return data


def split_by_best_threshold(data, target_attr, numeric_attr):
    # Calcula el mejor threshold para el atributo numérico numeric_attr
    # midiendo según la gananacia de información.

    thresholds = []

    # Se ordenan los ejemplos en orden ascendente según los valores de numeric_attr.
    sorted_data = sorted(data, key=lambda x: x[numeric_attr])
    
    # Se recorre el conjunto data de a 2 elementos para encontrar posibles
    # thresholds.
    for i in range(0, len(sorted_data) - 1):
        instance_1 = sorted_data[i]
        instance_2 = sorted_data[i + 1]

        # En caso de encontrar un posible candidato se almacena
        if instance_1[target_attr] != instance_2[target_attr] and instance_1[numeric_attr] != instance_2[numeric_attr]:
            thresholds.append(
                (instance_1[numeric_attr] + instance_2[numeric_attr]) / 2)
    
    # Se recorre la lista de posibles thresholds.
    for threshold in thresholds:

        # Se dividen los valores de numeric_attr según el threshold dado.
        splitted_data = split_data(copy.deepcopy(data), numeric_attr, threshold)
        
        # Se busca el threshold que maximiza la ganancia de información.
        maximum_thresholds_tied = []
        max_ig = -1
        ig = information_gain(splitted_data, numeric_attr, target_attr)
        if ig > max_ig:
            max_ig = ig
            maximum_thresholds_tied = []
            maximum_thresholds_tied.append(splitted_data)
        elif ig == max_ig:
            maximum_thresholds_tied.append(splitted_data)

    best_splitted_data = random.choice(maximum_thresholds_tied)

    # Se setean los valores del conjunto de datos según los resultados obtenidos
    # por el mejor threshold.
    for i in range(len(data)):
        data[i][numeric_attr] = best_splitted_data[i][numeric_attr]


def split_data(data, numeric_attr, threshold):
    # Divide los valores del atributo numérico numeric_attr según
    # el valor del threshold pasado por parámetro.

    new_key = numeric_attr + '>' + str(threshold)

    for instance in data:
        if instance[numeric_attr] > threshold:
            instance[numeric_attr] = 1
        else:
            instance[numeric_attr] = 0
        
    return data

def find_most_common_value_in_S(instances, target_attr):
    # Instances es un diccionario con key un valor del attributo y value un arreglo con la fila que contiene valor key
    # EJ: {Alta: [{'Dedicacion': 'Alta', 'Dificultad': 'Alta', 'Horario': 'Nocturno', 'Humedad': 'Media', 'Humor Docente': 'Bueno', 'Salva': 'Yes'}]}
    # Busca el valor más comun que toma el atributo en todo S
    # Si existe más de un valor más comun, se devuelve uno aleatorio entre ellos.
    maximum_keys_tied = []
    max_quantity = -1
    for key, value in instances.items():
        if key != '-':
            if len(value) > max_quantity:
                max_quantity = len(value)
                maximum_keys_tied = []
                maximum_keys_tied.append(key)
            elif len(value) == max_quantity:
                maximum_keys_tied.append(key)
    return random.choice(maximum_keys_tied)




def information_gain_missing_values_second_method(data, attr, target_attr):
    # Calcula la ganancia de información del atributo attr sobre el
    # conjunto data.

    data_subsets = {}
    data_information_gain = 0.0

    # Se divide el conjunto data en subconjuntos que tienen en común
    # el valor del atributo attr.
    for instance in data:
        if (instance[attr] in data_subsets):
            data_subsets[instance[attr]].append(instance)
        else:
            data_subsets[instance[attr]] = [instance]

    # Extension para valores faltantes primer metodo
    if '-' in data_subsets:
        data_without_miss_value_instance = copy.deepcopy(data)
        for instance in data_subsets['-']:
            data_without_miss_value_instance.remove(instance)
        for key, value in data_subsets.items():
            if key != '-':
                probability_subsets = {key: len(value)/len(data_without_miss_value_instance)}
    


    # Se calcula el valor de information gain según lo visto en teórico.
    data_information_gain = entropy(data, target_attr)
    for data_subset in data_subsets.values():
        data_information_gain -= (len(data_subset) / len(data)) * \
            entropy(data_subset, target_attr)
    return data_information_gain