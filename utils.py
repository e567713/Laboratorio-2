import math
from scipy.io import arff


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


def information_gain(data, target_attr):
    # Incompleto
    data_splitted = {}

    for instance in data:
        if (instance[target_attr] in data_splitted):
            data_splitted[instance[target_attr]].append(instance)
        else:
            data_splitted[instance[target_attr]] = [instance]

    return data_splitted


def read_file(path):
    return arff.loadarff(path)[0]
