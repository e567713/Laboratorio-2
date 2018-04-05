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
        best_attr_values = [instance[best_attr] for instance in data]

        # Se genera un árbol con el atributo best_attr en su raíz.
        tree = {
            'data': best_attr,
            'childs': {}
        }

        for value in best_attr_values:

            # Nos quedamos con los ejemplos de data que tengan el valor
            # value para el atributo best_attribute.
            filtered_data = [instance for instance in data if instance[best_attr] == value]

            # Si filtered_data es vacío → etiquetar con el valor más probable.
            if not filtered_data:
                tree['childs'][value] = {
                    'data': most_common(target_attr_values),
                    'childs': {}
                }
            
            # Si filtered_data no es vacío
            else:
                # Se quita a best_attr de la lista de atributos.
                filtered_attributes= copy.deepcopy(attributes)
                filtered_attributes.remove(best_attr)
                tree['childs'][value] = ID3_algorithm(filtered_data, filtered_attributes, target_attr)
        return tree


def read_file(path):
    return arff.loadarff(path)[0]


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
    attr_max_ig = attributes[0]
    for attr in attributes:
        ig = information_gain(data, attr, target_attr)
        if ig > max_ig:
            max_ig = ig
            attr_max_ig = attr
            maximum_values_tied = []
            maximum_values_tied.append(attr)
        elif ig == max_ig:
            maximum_values_tied.append(attr)
    return random.choice(maximum_values_tied)

def 